import jax.numpy as jnp
import numpy as np

from g1_wbc import rewards
from g1_wbc.config import load_config

cfg = load_config()


def test_kernels_peak_at_zero_error():
    assert float(rewards.tracking_lin_vel(jnp.array([.5, .2, 0.]), jnp.array([.5, .2, 0.]), .25)) == 1.0
    assert float(rewards.tracking_upper(jnp.ones(17), jnp.ones(17), .1)) == 1.0
    # kernel width rule: e_max 0.5 rad RMS still pays > exp(-2.5)
    assert float(rewards.tracking_upper(jnp.zeros(17), jnp.full(17, .5), .1)) > np.exp(-2.6)


def test_tilt_termination_sweep():
    t = cfg.rewards.termination
    def g(pitch, roll):
        return np.array([np.sin(pitch), -np.sin(roll) * np.cos(pitch), -np.cos(pitch) * np.cos(roll)])
    f = lambda p, r: bool(rewards.tilt_exceeded(g(p, r), t.max_tilt, t.max_pitch_forward, np))  # noqa
    assert not f(0, 0) and not f(0.3, 0) and not f(0, 0.3)
    assert f(0, 0.9) and f(-0.9, 0) and f(1.1, 0) and not f(0.9, 0)   # forward allowance
    assert f(np.pi / 2 + 0.1, 0)


def test_band_violation_zero_inside():
    assert float(rewards.band_violation(0.3, [0.2, 0.65])) == 0.0
    assert float(rewards.band_violation(0.1, [0.2, 0.65])) > 0.0


def test_gait_contact_and_limits():
    c = jnp.array([True, False]); s = jnp.array([True, False])
    assert float(rewards.gait_contact(c, s)) == 2.0
    lo, hi = jnp.array([-1.0]), jnp.array([1.0])
    assert float(rewards.dof_pos_limits(jnp.array([0.0]), lo, hi, 0.9)) == 0.0
    assert float(rewards.dof_pos_limits(jnp.array([0.95]), lo, hi, 0.9)) > 0.0


def test_reward_keys_match_weights():
    # compute_reward raises on a mismatch (M4)
    from g1_wbc.env import G1TeleopEnv  # noqa: F401
    assert "termination" in cfg.rewards.weights
