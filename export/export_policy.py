#!/usr/bin/env python
"""Export a distilled student to the deployment contract
(hospital-sim docs/interfaces.md section 4): exports/<name>/policy.onnx + policy.json.

  ./run.sh export/export_policy.py --checkpoint outputs/checkpoints/<student>/ckpt_best \
        --name g1_teleop_v1 [--curriculum teleop]

Steps (skill Phase 8): flax params -> torch module (layer by layer) -> JAX vs
torch parity on 1000 plausible inputs -> ONNX (opset 17, static shapes,
normalisation baked in) -> onnxruntime vs torch parity. Fails loudly on any
parity miss (tol 1e-4). torch is CPU-only here and used ONLY for export.

ONNX I/O: input  "obs_history" float32 (1, history_len * frame_dim): RAW frames
          (policy.json "obs" layout) concatenated oldest first, newest last
          (g1_controller's history layout);
          output "action" float32 (1, 29) in [-1, 1].
"""
import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ACT = {"relu": torch.relu, "elu": nn.functional.elu, "swish": nn.functional.silu,
       "tanh": torch.tanh}


class StudentTorch(nn.Module):
    """history (B, H, D) raw -> normalise -> prior MLP -> mu_p -> decoder([s_t, mu_p]) -> tanh."""

    def __init__(self, H, D, vae, prior_sizes, dec_sizes, n_act, activation):
        super().__init__()
        self.register_buffer("mean", torch.zeros(D))
        self.register_buffer("std", torch.ones(D))
        self.H, self.D, self.vae, self.act = H, D, vae, ACT[activation]
        ps = [H * D] + list(prior_sizes) + [2 * vae]
        self.prior = nn.ModuleList(nn.Linear(ps[i], ps[i + 1]) for i in range(len(ps) - 1))
        ds = [D + vae] + list(dec_sizes) + [n_act]
        self.dec = nn.ModuleList(nn.Linear(ds[i], ds[i + 1]) for i in range(len(ds) - 1))

    def _mlp(self, layers, x):
        for lin in layers[:-1]:
            x = self.act(lin(x))
        return layers[-1](x)

    def forward(self, flat):
        # (B, H * D) flat, oldest frame first (g1_controller's layout)
        x = (flat.reshape(flat.shape[0], self.H, self.D) - self.mean) / self.std
        mu = self._mlp(self.prior, x.reshape(x.shape[0], self.H * self.D))[:, : self.vae]
        return torch.tanh(self._mlp(self.dec, torch.cat([x[:, -1], mu], dim=-1)))


def _copy(layers, p):
    for i, lin in enumerate(layers):
        lin.weight.data = torch.from_numpy(np.asarray(p[f"hidden_{i}"]["kernel"]).T.copy())
        lin.bias.data = torch.from_numpy(np.asarray(p[f"hidden_{i}"]["bias"]).copy())


def build_torch(ckpt, cfg):
    from g1_wbc.distill_student import StudentJax
    from g1_wbc.layout import N_ACT, OBS_DIM
    sj = StudentJax(ckpt, cfg)
    s = cfg.train.student
    m = StudentTorch(int(s.history_len), OBS_DIM, int(s.vae_dim), s.prior_hidden,
                     s.decoder_hidden, N_ACT, s.activation)
    m.mean.data = torch.from_numpy(sj.mean.astype(np.float32))
    m.std.data = torch.from_numpy(sj.std.astype(np.float32))
    p = sj.params["params"]
    _copy(m.prior, p["prior"])
    _copy(m.dec, p["decoder"])
    m.eval()
    return m, sj


def plausible_inputs(sj, H, n, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, H, sj.mean.shape[0])).astype(np.float32)
    return (sj.mean + 2.0 * sj.std * noise).astype(np.float32)


def _obs_entry(name, dim, cfg):
    """policy.json obs entry, with the extra keys g1_controller's builder reads."""
    e = {"name": name, "dim": dim, "scale": 1.0}
    if name in ("joint_pos_rel", "joint_vel"):
        e["joints"] = "all"
    if name == "gait_phase":
        e["period"] = float(cfg.commands.gait.period_s)
    return e


def sidecar(cfg, name, ckpt_meta, onnx_file, H):
    from g1_wbc.config import base_height_target, model_info, stage_commands
    from g1_wbc.layout import (JOINT_NAMES, N_ACT, OBS_TERMS, UPPER_IDX, UPPER_NAMES, effort,
                               torque_limit)
    import mujoco
    from g1_wbc.config import EVAL_XML
    act = cfg.actuators
    mi = model_info()
    mj = mujoco.MjModel.from_xml_path(EVAL_XML)
    lim = [[float(v) for v in mj.jnt_range[mj.joint(n).id]] for n in JOINT_NAMES]
    cur = ckpt_meta.get("curriculum", "teleop")
    final = cfg.stages(cur)[-1]
    c = stage_commands(cfg.commands, final)
    bh = cfg.commands.base_height
    defs = {
        "base_ang_vel": "pelvis IMU angular velocity, rad/s, pelvis frame (LowState.imu_state.gyroscope)",
        "projected_gravity": "R_pelvis^T (0,0,-1): gravity unit vector in the pelvis frame "
                             "(from LowState.imu_state.quaternion, wxyz)",
        "cmd_vel": "[vx m/s, vy m/s, wz rad/s] from /g1/cmd_vel; 0 when absent",
        "cmd_base_height": "commanded pelvis height minus home_pelvis_height (m); OPTIONAL and "
                           "default-off: send 0 unless commands.base_height.enabled",
        "upper_body_target_rel": "/g1/upper_body_target (17 joints, upper_body_joint_names "
                                 "order) minus default_qpos[12:29]; 0 when absent (default arms)",
        "joint_pos_rel": "q - default_qpos, all 29 joints in joint_names order "
                         "(LowState.motor_state[i].q)",
        "joint_vel": "qd, rad/s (LowState.motor_state[i].dq)",
        "last_action": "this policy's previous output (raw, in [-1, 1]); 0 at start",
        "gait_phase": "[sin(2 pi phase), cos(2 pi phase)], phase = (t mod gait.period_s) / "
                      "gait.period_s, t = seconds since the policy was (re)started "
                      "(same clock as the Unitree G1 placeholder policy)",
    }
    return {
        # ---- docs/interfaces.md section 4 (exact keys)
        "name": name,
        "control_dt": float(act.control.physics_dt) * int(act.control.decimation),
        "sim_dt": float(act.control.physics_dt),
        "joint_names": list(JOINT_NAMES),
        "controlled_joints": list(range(N_ACT)),
        "default_qpos": [float(v) for v in act.default_pose],
        "action_scale": [float(v) for v in act.action_scale],
        "kp": [float(v) for v in act.kp],
        "kd": [float(v) for v in act.kd],
        "torque_limit": [float(v) for v in torque_limit(act)],
        "obs": [_obs_entry(n, d, cfg) for n, d in OBS_TERMS],
        "history_len": H,
        "commands": {"cmd_vel": [3], "upper_body_target": [17], "base_height": [1]},
        "placeholder": False,
        "action_clip": 1.0,   # g1_controller clips the raw action to +-action_clip before scaling
        # ---- extra keys (additive; everything g1_controller needs beyond section 4)
        "onnx": {"file": os.path.basename(onnx_file), "input": "obs_history",
                 "input_shape": [1, H * sum(d for _, d in OBS_TERMS)], "output": "action",
                 "output_shape": [1, N_ACT], "opset": 17},
        "obs_definitions": defs,
        "history": {"order": "oldest_first_newest_last", "layout": "flat (1, H * frame_dim)",
                    "reset_fill": "first_obs",
                    "note": "on (re)start fill all history_len rows with the first frame "
                            "(ULTRA convention); normalisation is inside the graph, feed RAW"},
        "action_rule": "q_target = clip(default_qpos + clip(action, -1, 1) * action_scale, "
                       "joint_limits); tau = clip(kp (q_target - q) - kd qd, +-torque_limit), "
                       "at sim_dt (500 Hz) or as LowCmd q/dq=0/kp/kd/tau=0",
        "joint_limits": lim,
        "upper_body_joint_names": list(UPPER_NAMES),
        "upper_body_joint_indices": list(UPPER_IDX),
        "gait": dict(cfg.commands.gait),
        "home_pelvis_height": float(base_height_target(cfg)),
        "command_details": {
            "cmd_vel": {"trained_range": {"vx": list(c.vx_range), "vy": list(c.vy_range),
                                          "wz": list(c.wz_range)},
                        "zero_below_xy": float(cfg.commands.cmd_vel.zero_below)},
            "upper_body_target": {"trained_envelope": "configs/upper_body_poses.yaml",
                                  "max_joint_speed_rad_s": float(cfg.commands.upper_body.joint_speed_max),
                                  "only_when": "cmd_vel == 0 (MANIP)"},
            "base_height": {"enabled": bool(bh.enabled),
                            "trained_range_offset_m": [float(v) for v in bh.offset_range]},
        },
        "training": {"curriculum": cur, "final_stage": final.name,
                     "teacher_ckpt": os.path.basename(ckpt_meta.get("teacher_ckpt", "")),
                     "student_iteration": ckpt_meta.get("iteration"),
                     "domain_rand": "configs/domain_rand.yaml (ULTRA ranges)",
                     "obs_noise_uniform_amplitude": {k: v for k, v in cfg.domain_rand.obs_noise.items()
                                                      if k != "distribution"}},
        "robot": {"model_source": mi.source_name, "fallback_model": bool(mi.fallback),
                  "effort_source": act.effort_source, "effort": effort(act),
                  "armature": [float(v) for v in act.armature],
                  "mass_kg": mi.train.info.total_mass_kg},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    import onnxruntime as ort
    from g1_wbc.config import RL_ROOT, load_config
    cfg = load_config()
    H = int(cfg.train.student.history_len)
    model, sj = build_torch(a.checkpoint, cfg)
    x = plausible_inputs(sj, H, 1000, 0)
    xf = x.reshape(x.shape[0], -1)
    with torch.no_grad():
        t_out = model(torch.from_numpy(xf)).numpy()
    j_out = np.stack([sj(h) for h in x[:200]])
    d_jt = float(np.max(np.abs(t_out[:200] - j_out)))
    assert d_jt < a.tol, f"JAX vs torch parity FAILED {d_jt:.2e}"
    out_dir = os.path.join(a.out_root or os.path.join(RL_ROOT, "exports"), a.name)
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "policy.onnx")
    torch.onnx.export(model, (torch.zeros(1, H * x.shape[-1]),), onnx_path, opset_version=17,
                      input_names=["obs_history"], output_names=["action"], dynamo=False)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    d_ot = max(float(np.max(np.abs(sess.run(None, {"obs_history": xf[i:i + 1]})[0] - t_out[i:i + 1])))
               for i in range(0, 1000, 5))
    assert d_ot < a.tol, f"onnxruntime vs torch parity FAILED {d_ot:.2e}"
    meta = sj.meta
    sc = sidecar(cfg, a.name, meta, onnx_path, H)
    sc["export_parity"] = {"jax_vs_torch": d_jt, "onnxruntime_vs_torch": d_ot}
    with open(os.path.join(out_dir, "policy.json"), "w") as f:
        json.dump(sc, f, indent=2)
    print(f"parity: jax-torch {d_jt:.1e}, ort-torch {d_ot:.1e}\n-> {onnx_path}\n-> "
          f"{os.path.join(out_dir, 'policy.json')}")


if __name__ == "__main__":
    main()
