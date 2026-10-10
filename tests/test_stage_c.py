import os

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from g1_wbc.config import MJX_XML, RL_ROOT, _deep_merge, load_config
from g1_wbc.motion_data import MotionLib, forward_kinematics

SYN = os.path.join(RL_ROOT, "data", "motions_synthetic")
pytestmark = pytest.mark.skipif(not os.path.isdir(SYN), reason="run scripts/make_synthetic_motion.py")


def test_motion_lib_and_fk():
    lib = MotionLib(SYN)
    cfg = load_config("stage_c_tracking.yaml")
    m = mujoco.MjModel.from_xml_path(MJX_XML)
    r = {"root_pos": lib.root_pos[0, :3], "root_quat": lib.root_quat[0, :3], "dof": lib.dof[0, :3]}
    assert np.allclose(forward_kinematics(m, r, cfg.train.tracking.key_bodies), lib.key_pos[0, :3], atol=1e-5)
    assert lib.fps == 50.0


def test_tracking_env_reset_step():
    from g1_wbc.tracking_env import TRACK_OBS, G1TrackingEnv
    cfg = load_config("stage_c_tracking.yaml")
    cfg.train = _deep_merge(cfg.train, {"tracking": {"motion_dir": "data/motions_synthetic",
                                                     "ref_root_scale": 1.0}})
    env = G1TrackingEnv(cfg, cfg.stages("tracking")[0])
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert s.obs["state"].shape == (TRACK_OBS,)
    step = jax.jit(env.step)
    for _ in range(3):
        s = step(s, jnp.zeros(29))
    assert np.isfinite(float(s.reward)) and float(s.reward) > 0
    # standing at home on a standing-start reference: feet error ~0
    assert float(s.metrics["reward/feet_err_m"]) < 0.05
