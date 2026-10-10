"""Env semantics on CPU JAX (compiles once per module)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from g1_wbc import commands as C
from g1_wbc.config import load_config, stage_domain_rand
from g1_wbc.env import G1TeleopEnv
from g1_wbc.layout import OBS_DIM, PRIV_DIM, UPPER_IDX, obs_slice

cfg = load_config()
FINAL = cfg.stages("teleop")[-1]


@pytest.fixture(scope="module")
def env():
    return G1TeleopEnv(cfg, FINAL, student_noise=True, emit_clean_state=True)


@pytest.fixture(scope="module")
def batch(env):
    r = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(3), 256))
    return r


def test_shapes(batch):
    assert batch.obs["state"].shape == (256, OBS_DIM)
    assert batch.obs["privileged_state"].shape == (256, PRIV_DIM)


def test_privileged_equals_sampled_latents(batch):
    """L14: every privileged slot equals the episode's sampled value."""
    p, i = np.asarray(batch.obs["privileged_state"]), batch.info
    assert np.allclose(p[:, 0], i["friction"])
    assert np.allclose(p[:, 1] * 3.0, i["torso_mass_delta"])
    assert np.allclose(p[:, 5:34], i["kp_scale"]) and np.allclose(p[:, 34:63], i["kd_scale"])
    assert np.allclose(p[:, 63], i["action_delay"])


def test_dr_varies_and_hits_model(batch, env):
    """M16: DR changes the model fields across episodes."""
    f = batch.info["model_fields"]
    assert np.std(np.asarray(f["body_mass"])[:, env._torso_id]) > 0.5
    assert np.std(np.asarray(f["geom_friction"])[:, env._floor, 0]) > 0.2
    assert set(np.unique(np.asarray(batch.info["action_delay"]))) == {0, 1}


def test_student_noise_is_uniform_bounded_teacher_clean(batch, env):
    d = np.asarray(batch.obs["state"] - batch.obs["state_clean"])
    amp = np.asarray(env._noise_amp)
    assert np.all(np.abs(d) <= amp + 1e-6) and np.abs(d).max() > 0
    teacher = G1TeleopEnv(cfg, FINAL)
    assert float(jnp.max(teacher._noise_amp)) == 0.0          # L15


def test_modes_and_commands(batch):
    m1, m2 = np.asarray(batch.info["mode1"]), np.asarray(batch.info["mode2"])
    legal = {(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 0), (2, 0)}
    assert set(zip(m1.tolist(), m2.tolist())) <= legal      # never loco<->manip
    cmd = np.asarray(batch.obs["state_clean"][:, obs_slice("cmd_vel")])
    assert np.all(cmd[m1 != C.LOCO] == 0.0)
    ub = np.asarray(batch.obs["state_clean"][:, obs_slice("upper_body_target_rel")])
    assert np.allclose(ub[m1 != C.MANIP], 0.0)


def test_upper_schedule_smooth_and_within_limits(env):
    info = {"mode1": jnp.int32(C.MANIP), "mode2": jnp.int32(C.STAND),
            "switch_step": jnp.int32(300), **C.init_upper(None, env._bank, None, C.MANIP, 1.0, env.dt)}
    adv = jax.jit(lambda i, s, k: C.advance_upper(i, s, k, env._bank, cfg.commands.upper_body, 1.0, env.dt))
    key, prev, maxstep = jax.random.PRNGKey(0), None, 0.0
    for s in range(400):
        key, k = jax.random.split(key)
        info = adv(info, jnp.int32(s), k)
        t = np.asarray(info["ub_target"])
        assert np.all(t >= np.asarray(env._bank.lo) - 1e-6) and np.all(t <= np.asarray(env._bank.hi) + 1e-6)
        if prev is not None:
            maxstep = max(maxstep, np.abs(t - prev).max())
        prev = t
    # joint_speed_max 2 rad/s, min-jerk peak speed 1.875 x mean -> <= ~0.08 rad/step
    assert maxstep < 0.1
    assert np.allclose(t, np.asarray(env._bank.default), atol=1e-5)   # back to default after MANIP


def test_gait_clock_matches_unitree():
    obs, stance = C.gait_clock(jnp.int32(10), 0.02, cfg.commands.gait)   # t = 0.2 s
    ph = (0.2 % 0.8) / 0.8
    assert np.allclose(obs, [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)], atol=1e-6)
    assert list(np.asarray(stance)) == [ph < 0.55, (ph + 0.5) % 1 < 0.55]


def test_step_runs_and_standing_reward_positive(env):
    st = cfg.stages("stand")[0]
    e = G1TeleopEnv(cfg, st)
    s = jax.jit(e.reset)(jax.random.PRNGKey(0))
    s2 = jax.jit(e.step)(s, jnp.zeros(29))
    assert float(s2.reward) > 0 and float(s2.done) == 0.0
    assert set(k[7:] for k in s2.metrics if k.startswith("reward/")) == set(e.reward_keys())
