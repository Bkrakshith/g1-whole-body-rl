"""Evaluate ULTRA's released tracking teacher in MuJoCo on ULTRA's G1 and on our G1 + BrainCo Revo2.

Reuses ULTRA's own sim2sim code unchanged (observation builder, motion and object loading, reset, policy
network and checkpoint, 60 Hz policy / 1020 Hz PD loop, gains, action scaling) and swaps only the robot file:

  --robot ultra            ULTRA's g1_29dof.xml (rubber hands), ULTRA torque clip 0.8 x 50 Nm ankles (baseline)
  --robot ours --effort ultra   our G1 29-DoF rev 1.0 + BrainCo Revo2, ankle/waist-roll-pitch 50 Nm
  --robot ours --effort model   the same robot at its 35 Nm limits (the limits chosen for this project)

Modes:
  track (default)  each clip is tracked to its end, then the last frame is held for as long again (stand phase)
  stand            the first frame of each clip is held for --duration s: standing still next to the object,
                   with optional --push N horizontal pelvis pushes (0.2 s each, random direction)

Metrics per clip: fall (pelvis below 0.45 m, or mean body-position error above 0.5 m), mean body-position
error over ULTRA's 39 tracked bodies (MPBPE), root and joint errors, object position error, torque saturation.

Run with ultra_eval/run.sh (conda env ultra-sim2sim, pinned to the good CPUs):
    ultra_eval/run.sh eval_teacher.py --robot ours --effort model --clips 40 --workers 12
"""
import argparse
import glob
import json
import os
import random
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)                  # g1-whole-body-rl
REPO = os.path.dirname(RL)                  # hospital-sim master checkout (datasets/ lives here)
ULTRA = os.path.join(RL, "third_party", "ULTRA")
MOTIONS = os.path.join(REPO, "datasets", "ultra", "OMOMO_retarget_aug_v4")
CKPT = os.path.join(ULTRA, "ultra", "weights", "teacher_ultra_inference.pth")
ROBOTS = {
    ("ultra", "ultra"): os.path.join(ULTRA, "ultra", "data", "assets", "g1", "g1_29dof.xml"),
    ("ours", "ultra"): os.path.join(HERE, "assets", "g1_brainco_ultra_ultra.xml"),
    ("ours", "model"): os.path.join(HERE, "assets", "g1_brainco_ultra_model.xml"),
}
# ULTRA sim2sim_teacher.py constants (copied verbatim: gains, armature-derived; ULTRA effort; default pose)
KP = np.array([40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
               40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
               40.179238, 28.501246, 28.501246,
               14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
               14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327])
KD = np.array([2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
               2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
               2.557890, 1.814446, 1.814446,
               0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
               0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142])
ULTRA_EFFORT = np.array([88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50, 88, 50, 50,
                         25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], dtype=float)
SIM_DT, DECIMATION, ACTION_SCALE = 1.0 / (60.0 * 17.0), 17, 3.0
FALL_Z, FALL_MPBPE = 0.45, 0.5
ALIASES = {"left_rubber_hand": "left_", "right_rubber_hand": "right_"}   # hand contact = palm or any finger

_W = {}


def _setup_worker():
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, os.path.join(ULTRA, "ultra"))
    os.chdir(ULTRA)  # ULTRA's loaders use repo-relative asset paths
    import functools
    import torch
    torch.set_num_threads(1)
    torch.load = functools.partial(torch.load, map_location="cpu", weights_only=False)
    import sim2sim_teacher as st
    from utils import obs as uobs
    _W.update(torch=torch, st=st, uobs=uobs, orig_contact=uobs.contact_indicator_for_bodies)
    _W["policy"], _W["mean"], _W["var"] = _load_policy()


def _load_policy():
    """ULTRA's own network build + checkpoint load (sim2sim_teacher.HumanoidEnv.__init__, non-JIT branch)."""
    from rl_games.algos_torch import torch_ext
    from learning import ultra_network_builder, ultra_models
    params = {"name": "ultra", "separate": True,
              "space": {"continuous": {"mu_activation": "None", "sigma_activation": "None",
                                       "mu_init": {"name": "default"},
                                       "sigma_init": {"name": "const_initializer", "val": -2.9},
                                       "fixed_sigma": True, "learn_sigma": False}},
              "mlp": {"units": [1024, 1024, 512], "activation": "relu", "d2rl": False,
                      "initializer": {"name": "default"}, "regularizer": {"name": "None"}}}
    builder = ultra_network_builder.UltraBuilder()
    builder.load(params)
    net = ultra_models.ModelUltraContinuous(builder)
    ck = torch_ext.load_checkpoint(CKPT)
    policy = net.build({"actions_num": 29, "input_shape": (4052,), "num_seqs": 1, "value_size": 1})
    policy.load_state_dict(ck["model"])
    policy.eval()
    rms = ck["running_mean_std"]
    return policy, rms["running_mean"].float(), rms["running_var"].float()


def _patch_contacts(model):
    """In our model the BrainCo palm is fused into wrist_yaw_link and the fingers are separate bodies, so the
    policy's rubber_hand contact flag is set by contact anywhere on that hand."""
    import mujoco
    uobs = _W["uobs"]
    orig = _W["orig_contact"]
    groups = {}
    for marker, side in ALIASES.items():
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, marker)
        wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}wrist_yaw_link")
        members = {wrist} | {b for b in range(model.nbody)
                             if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(side)
                             and any(k in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                                     for k in ("thumb", "index", "middle", "ring", "pinky"))}
        groups[mid] = members

    def contact_indicator(model_, data, body_ids, device=None):
        ind = orig(model_, data, body_ids, device)
        if not groups:
            return ind
        touching = set()
        for i in range(data.ncon):
            c = data.contact[i]
            touching.add(int(model_.geom_bodyid[c.geom1]))
            touching.add(int(model_.geom_bodyid[c.geom2]))
        for j, bid in enumerate(body_ids):
            if int(bid) in groups and groups[int(bid)] & touching:
                ind[0, j] = 1.0
        return ind
    uobs.contact_indicator_for_bodies = contact_indicator


def run_clip(job):
    import mujoco
    torch, st = _W["torch"], _W["st"]
    robot_xml, motion, mode, cfg = job["robot_xml"], job["motion"], job["mode"], job["cfg"]
    rng = np.random.default_rng(job["seed"])

    env = object.__new__(st.HumanoidEnv)
    env.device = "cpu"
    object_name = os.path.basename(motion).split("_")[1]
    object_xml = f"ultra/data/assets/objects/{object_name}.xml"
    env._load_object(object_name, object_xml)
    env._load_motion(motion)
    n_orig = env.hoi_data.shape[0] // 2          # _load_motion doubles the clip, holding the last frame
    if mode == "stand":
        n_hold = int(cfg["duration"] * 60) + 2
        for key in ("hoi_data", "hoi_ref"):
            t = getattr(env, key)
            setattr(env, key, t[:1].repeat(n_hold, 1))
        for k, v in list(env.hoi_data_dict.items()):
            if hasattr(v, "shape") and v.shape[0] == 2 * n_orig:
                env.hoi_data_dict[k] = v[:1].repeat(n_hold, 1)
        env.max_episode_length = n_hold
        n_orig = 0

    with tempfile.TemporaryDirectory() as tmp:
        merged = st.merge_mjcf(robot_xml, object_xml, os.path.join(tmp, "scene.xml"))
        model = mujoco.MjModel.from_xml_path(merged)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    env.model, env.data, env.object_name = model, data, object_name
    if job["robot"] == "ours":
        _patch_contacts(model)
    else:
        _W["uobs"].contact_indicator_for_bodies = _W["orig_contact"]
    env.obs_builder = st.MujocoObs(model, object_name, env.max_episode_length, env.hoi_data,
                                   env.object_points, history_step=10)
    env.reset_human_and_object_from_ref()

    effort = model.actuator_ctrlrange[:29, 1].copy() if job["robot"] == "ours" else ULTRA_EFFORT
    tlim = np.minimum(0.8 * ULTRA_EFFORT, effort)      # ULTRA's 0.8 clip, never above the robot's motor limit
    ob = env.obs_builder
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if obj_body < 0:
        obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{object_name}_obj")
    ref = env.hoi_data_dict
    pushes = sorted(rng.uniform(3.0, cfg["duration"] - 2.0, cfg.get("push", 0))) if mode == "stand" else []
    push_dir = [rng.uniform(0, 2 * np.pi) for _ in pushes]

    renderer = writer = None
    if job.get("video"):
        import imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type, cam.trackbodyid, cam.distance, cam.elevation, cam.azimuth = \
            mujoco.mjtCamera.mjCAMERA_TRACKING, pelvis, 3.2, -15, 140
        writer = imageio.get_writer(job["video"], fps=30, codec="libx264", quality=7, macro_block_size=None)

    last_action = torch.zeros((1, 29))
    last_torque = torch.zeros((1, 29))
    errs = {"mpbpe": [], "root_xy": [], "root_z": [], "dof": [], "obj": []}
    sat, steps, fell, fell_t, fall_reason = 0, 0, False, None, ""
    n_steps = (env.max_episode_length - 1) * DECIMATION
    pd_target = np.zeros(29)
    t0 = time.perf_counter()
    for i in range(n_steps):
        dof_pos, dof_vel = st._gather_dof_slices(data, ob.dof_qpos_ids, ob.dof_qvel_ids)
        dof_pos, dof_vel = dof_pos.squeeze(0).numpy(), dof_vel.squeeze(0).numpy()
        if i % DECIMATION == 0:
            ts = i // DECIMATION
            obs = ob._compute_observations(data, ts, last_action, last_torque, student_obs=False, episode_length=ts)
            with torch.no_grad():
                x = torch.clamp((obs - _W["mean"]) / torch.sqrt(_W["var"] + 1e-5), -5.0, 5.0)
                mu = _W["policy"]({"is_train": False, "prev_actions": None, "obs": x, "rnn_states": None})["mus"]
                action = torch.clamp(mu, -1.0, 1.0).squeeze(0).numpy()
            last_action = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
            pd_target = action * ACTION_SCALE
            # metrics against the reference frame the policy is tracking now
            body = np.stack([data.xpos[b] for b in ob.body_ids])
            rb = ref["body_pos"][ts].numpy().reshape(-1, 3)
            mpbpe = float(np.linalg.norm(body - rb, axis=1).mean())
            errs["mpbpe"].append((ts, mpbpe))
            errs["root_xy"].append((ts, float(np.linalg.norm(data.xpos[pelvis][:2] - ref["root_pos"][ts, :2].numpy()))))
            errs["root_z"].append((ts, float(abs(data.xpos[pelvis][2] - ref["root_pos"][ts, 2].item()))))
            errs["dof"].append((ts, float(np.abs(dof_pos - ref["dof_pos"][ts].numpy()).mean())))
            if obj_body >= 0:
                errs["obj"].append((ts, float(np.linalg.norm(data.xpos[obj_body] - ref["obj_pos"][ts].numpy()))))
            if not fell and (data.xpos[pelvis][2] < FALL_Z or mpbpe > FALL_MPBPE):
                fell, fell_t = True, ts / 60.0
                fall_reason = "pelvis low" if data.xpos[pelvis][2] < FALL_Z else "tracking error > 0.5 m"
            if writer is not None and ts % 2 == 0:
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
            if fell and not job.get("video"):
                break
        data.xfrc_applied[pelvis, :] = 0.0
        t_now = data.time
        for tp, ang in zip(pushes, push_dir):
            if tp <= t_now < tp + 0.2:
                data.xfrc_applied[pelvis, 0:2] = cfg["push_force"] * np.array([np.cos(ang), np.sin(ang)])
        torque = (pd_target - dof_pos) * KP - dof_vel * KD
        sat += int(np.sum(np.abs(torque) > tlim))
        torque = np.clip(torque, -tlim, tlim)
        data.ctrl[:29] = torque
        last_torque = torch.as_tensor(torque, dtype=torch.float32).unsqueeze(0)
        mujoco.mj_step(model, data)
        steps += 1
    if writer is not None:
        writer.close()
        renderer.close()

    def mean(key, lo=0, hi=None):
        v = [e for t, e in errs[key] if t >= lo and (hi is None or t < hi)]
        return float(np.mean(v)) if v else None
    out = {"clip": os.path.basename(motion), "object": object_name, "mode": mode, "fell": fell,
           "fell_at_s": fell_t, "fall_reason": fall_reason,
           "seconds": len(errs["mpbpe"]) / 60.0, "wall_s": round(time.perf_counter() - t0, 1),
           "torque_saturation": sat / max(1, steps * 29)}
    for key in errs:
        out[f"{key}_all"] = mean(key)
        if mode == "track":
            out[f"{key}_motion"] = mean(key, 0, n_orig)
            out[f"{key}_hold"] = mean(key, n_orig)
    if pushes:
        out["pushes_s"] = [round(p, 2) for p in pushes]
    return out


def summarise(results):
    n = len(results)
    falls = sum(r["fell"] for r in results)
    keys = [k for k in results[0] if k.endswith(("_all", "_motion", "_hold")) or k == "torque_saturation"]
    ok = [r for r in results if not r["fell"]] or results
    s = {"clips": n, "falls": falls, "survival": (n - falls) / n}
    for k in keys:
        vals = [r[k] for r in ok if r.get(k) is not None]
        s[k + "_mean_survivors"] = float(np.mean(vals)) if vals else None
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["ultra", "ours"], default="ours")
    ap.add_argument("--effort", choices=["model", "ultra"], default="model")
    ap.add_argument("--mode", choices=["track", "stand"], default="track")
    ap.add_argument("--clips", type=int, default=20, help="random clips (same seed -> same clips for every robot)")
    ap.add_argument("--clip", action="append", default=[], help="specific clip file name(s) instead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=20.0, help="stand mode: seconds")
    ap.add_argument("--push", type=int, default=0, help="stand mode: number of pushes")
    ap.add_argument("--push-force", type=float, default=150.0, help="N, applied for 0.2 s")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--video", default="", help="write an MP4 of the first clip here")
    ap.add_argument("--out", default="", help="JSON results file")
    args = ap.parse_args()
    if args.robot == "ultra":
        args.effort = "ultra"

    files = sorted(glob.glob(os.path.join(MOTIONS, "*.pt")))
    if args.clip:
        clips = [os.path.join(MOTIONS, c) for c in args.clip]
    else:
        clips = random.Random(args.seed).sample(files, args.clips)
    cfg = {"duration": args.duration, "push": args.push, "push_force": args.push_force}
    jobs = [{"robot": args.robot, "robot_xml": ROBOTS[(args.robot, args.effort)], "motion": c, "mode": args.mode,
             "cfg": cfg, "seed": args.seed * 1000 + k, "video": args.video if k == 0 else ""}
            for k, c in enumerate(clips)]
    jobs = [{**j, "video": os.path.abspath(j["video"])} if j["video"] else j for j in jobs]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs)), initializer=_setup_worker) as pool:
        results = []
        for r in pool.imap_unordered(run_clip, jobs):
            results.append(r)
            print(f"[{len(results)}/{len(jobs)}] {r['clip']:<52} {'FELL ' + r['fall_reason'] if r['fell'] else 'ok':<28}"
                  f" mpbpe {r['mpbpe_all']:.3f} m  sat {r['torque_saturation']:.3f}  ({r['wall_s']} s)", flush=True)
    summary = summarise(results)
    label = f"{args.robot}_{args.effort}_{args.mode}" + (f"_push{args.push}x{args.push_force:g}N" if args.push else "")
    print(json.dumps({label: summary}, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"config": {**vars(args), "label": label}, "summary": summary,
                   "results": sorted(results, key=lambda r: r["clip"])}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
