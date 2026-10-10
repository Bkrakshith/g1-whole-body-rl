"""Export -> sidecar -> (sim2sim, real g1_controller runner) agree; the MJX
training obs and the CPU evaluator obs agree for the same state (instrument
check, skill 'verify the instrument')."""
import importlib.util
import json
import os
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs

from g1_wbc import sim2sim
from g1_wbc.config import MJX_XML, RL_ROOT, load_config
from g1_wbc.layout import N_ACT, OBS_DIM, PRIV_DIM
from g1_wbc.networks import make_student
from g1_wbc.ppo_teacher import save_checkpoint

cfg = load_config()
sys.path.insert(0, os.path.join(RL_ROOT, "export"))
import export_policy as E  # noqa: E402

CONTRACT_KEYS = {"name", "control_dt", "sim_dt", "joint_names", "controlled_joints", "default_qpos",
                 "action_scale", "kp", "kd", "torque_limit", "obs", "history_len", "commands",
                 "placeholder"}
CTRL = os.path.join(RL_ROOT, "..", "ros2_ws", "src", "hospital-sim-ros2", "g1_controller", "g1_controller", "policy.py")


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("exp")
    H = int(cfg.train.student.history_len)
    st = running_statistics.init_state({"state": specs.Array((OBS_DIM,), jnp.float32),
                                        "privileged_state": specs.Array((PRIV_DIM,), jnp.float32)})
    k = jax.random.PRNGKey(0)
    st = running_statistics.update(st, {"state": jax.random.normal(k, (64, OBS_DIM)),
                                        "privileged_state": jax.random.normal(k, (64, PRIV_DIM))})
    stu = make_student(cfg, N_ACT)
    params = stu.init(k, jnp.zeros((1, H, OBS_DIM)), jnp.zeros((1, OBS_DIM + PRIV_DIM)),
                      jnp.zeros((1, int(cfg.train.student.vae_dim))))
    ck = str(tmp / "ckpt")
    save_checkpoint(ck, {"normalizer": st, "student": params}, {"curriculum": "teleop"})
    sys.argv = ["x", "--checkpoint", ck, "--name", "t", "--out_root", str(tmp)]
    E.main()
    return str(tmp / "t")


def test_sidecar_has_contract_keys(export_dir):
    sc = json.load(open(os.path.join(export_dir, "policy.json")))
    assert CONTRACT_KEYS <= set(sc)
    assert len(sc["joint_names"]) == 29 and len(sc["kp"]) == 29 and sc["placeholder"] is False
    assert sc["control_dt"] == 0.02 and sc["sim_dt"] == 0.002
    assert sum(o["dim"] for o in sc["obs"]) * sc["history_len"] == sc["onnx"]["input_shape"][1]
    assert sc["export_parity"]["onnxruntime_vs_torch"] < 1e-4


@pytest.mark.skipif(not os.path.exists(CTRL), reason="g1_controller not present")
def test_real_controller_builds_the_same_obs_and_targets(export_dir):
    spec = importlib.util.spec_from_file_location("ctrl_policy", CTRL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    runner = mod.PolicyRunner(export_dir)
    pol = sim2sim.OnnxPolicy(export_dir)
    sim = sim2sim.Sim2Sim(pol.sc, MJX_XML, cfg)
    rng = np.random.default_rng(0)
    d = sim.d
    mujoco.mj_resetDataKeyframe(sim.m, d, 0)
    d.qpos[7:] += rng.uniform(-0.2, 0.2, 29)
    q = rng.normal(size=4); d.qpos[3:7] = q / np.linalg.norm(q)
    d.qvel[:] = rng.normal(size=d.qvel.shape)
    mujoco.mj_forward(sim.m, d)
    cmd, ub = np.array([0.3, -0.1, 0.5]), sim.q0[sim.upper_idx] + rng.uniform(-0.3, 0.3, 17)
    a_prev = rng.uniform(-1, 1, 29).astype(np.float32)
    mine = sim.frame(0.3, cmd, ub, 0.0, a_prev)
    runner.last_action, runner.t = a_prev, 0.3
    theirs = runner.build_obs(d.qpos[7:].copy(), d.qvel[6:].copy(), d.qpos[3:7].copy(),
                              d.qvel[3:6].copy(), cmd, ub)
    assert np.allclose(mine, theirs, atol=1e-5)


def test_mjx_training_obs_equals_cpu_evaluator_obs():
    from g1_wbc.env import G1TeleopEnv
    env = G1TeleopEnv(cfg, cfg.stages("stand")[0], emit_clean_state=True)
    s = jax.jit(env.reset)(jax.random.PRNGKey(1))
    sc = E.sidecar(cfg, "x", {}, "x.onnx", 10)
    sim = sim2sim.Sim2Sim(sc, MJX_XML, cfg)
    sim.d.qpos[:] = np.asarray(s.pipeline_state.qpos)
    sim.d.qvel[:] = np.asarray(s.pipeline_state.qvel)
    mujoco.mj_forward(sim.m, sim.d)
    f = sim.frame(0.0, np.zeros(3), sim.q0[sim.upper_idx], 0.0, np.zeros(29))
    assert np.allclose(f, np.asarray(s.obs["state_clean"]), atol=1e-5)


def test_mjx_and_cpu_physics_agree_on_zero_action_fall():
    """Same PD hold in MJX (training) and CPU MuJoCo on the training model:
    pelvis height trajectories must agree (instrument check)."""
    from g1_wbc.env import G1TeleopEnv
    env = G1TeleopEnv(cfg, cfg.stages("stand")[0])
    s = jax.jit(env.reset)(jax.random.PRNGKey(2))
    step = jax.jit(env.step)
    m = mujoco.MjModel.from_xml_path(MJX_XML)
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(s.pipeline_state.qpos); d.qvel[:] = 0
    kp, kd = np.asarray(cfg.actuators.kp), np.asarray(cfg.actuators.kd)
    q0 = np.asarray(cfg.actuators.default_pose)
    for k in range(25):     # 0.5 s
        s = step(s, jnp.zeros(29))
        for _ in range(10):
            d.ctrl[:] = np.clip(kp * (q0 - d.qpos[7:]) - kd * d.qvel[6:],
                                m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            mujoco.mj_step(m, d)
    assert abs(float(s.pipeline_state.qpos[2]) - d.qpos[2]) < 0.01
