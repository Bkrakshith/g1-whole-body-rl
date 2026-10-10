"""Teleop command generator (Stage A/B): operator modes, velocity commands,
upper-body target trajectories. Pure JAX functions of state.info so the whole
schedule lives inside the jitted env step.

Modes mirror docs/interfaces.md 3.2 (the controller-side modes):
  STAND = 0  cmd_vel 0, upper body at default
  LOCO  = 1  cmd_vel from the teleop envelope, upper body at default
  MANIP = 2  cmd_vel 0, upper body follows smooth teleop trajectories
Legal in-episode switches only: STAND<->LOCO, STAND<->MANIP (LOCO->MANIP is
rejected by the manager); leaving MANIP blends the arms back to default over
return_to_default_s (contract: ~1 s).
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from .config import RL_ROOT
from .layout import UPPER_IDX

STAND, LOCO, MANIP = 0, 1, 2
BIG = jnp.iinfo(jnp.int32).max // 2


class UpperBodyBank:
    """Static arrays the sampler needs (numpy -> jnp constants)."""

    def __init__(self, commands_cfg, default_pose, joint_lo, joint_hi):
        ub = commands_cfg.upper_body
        with open(os.path.join(RL_ROOT, ub.pose_file)) as f:
            poses = yaml.safe_load(f)
        self.library = jnp.asarray(np.array([p["q"] for p in poses["library"]], np.float32))
        self.box_lo = jnp.asarray(poses["box_lo"], jnp.float32)
        self.box_hi = jnp.asarray(poses["box_hi"], jnp.float32)
        self.default = jnp.asarray(np.asarray(default_pose)[list(UPPER_IDX)], jnp.float32)
        self.lo = jnp.asarray(np.asarray(joint_lo)[list(UPPER_IDX)], jnp.float32)
        self.hi = jnp.asarray(np.asarray(joint_hi)[list(UPPER_IDX)], jnp.float32)
        bank_path = os.path.join(RL_ROOT, ub.motion_bank)
        self.p_bank = float(ub.p_motion_bank) if os.path.exists(bank_path) else 0.0
        if self.p_bank > 0:
            self.bank = jnp.asarray(np.load(bank_path)["upper_q"], jnp.float32)
        else:
            self.bank = self.library[:1]


def sample_pose(rng, bank: UpperBodyBank, ub_cfg, arm_scale):
    """One upper-body waypoint (17,): a feasible library pose + uniform noise,
    a uniform sample in the feasible box, or (if built) a frame of the ULTRA
    motion bank; blended toward default by the curriculum arm_scale."""
    k_src, k_idx, k_noise, k_box, k_bank = jax.random.split(rng, 5)
    lib = bank.library[jax.random.randint(k_idx, (), 0, bank.library.shape[0])]
    lib = lib + jax.random.uniform(k_noise, lib.shape, minval=-1.0, maxval=1.0) * ub_cfg.noise_rad
    box = jax.random.uniform(k_box, bank.box_lo.shape, minval=bank.box_lo, maxval=bank.box_hi)
    frame = bank.bank[jax.random.randint(k_bank, (), 0, bank.bank.shape[0])]
    u = jax.random.uniform(k_src)
    pose = jnp.where(u < bank.p_bank, frame,
                     jnp.where(u < bank.p_bank + (1 - bank.p_bank) * ub_cfg.p_library, lib, box))
    pose = bank.default + arm_scale * (pose - bank.default)
    return jnp.clip(pose, bank.lo, bank.hi)


def min_jerk(u):
    u = jnp.clip(u, 0.0, 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def sample_cmd_vel(rng, c, zero_below, p_zero_axis):
    """(3,) vx, vy, wz uniform in the stage's ranges; each axis zeroed with
    p_zero_axis (pure forward / strafe / turn commands, like key presses);
    |v_xy| < zero_below -> 0 (legged_gym convention)."""
    k = jax.random.split(rng, 4)
    v = jnp.array([
        jax.random.uniform(k[0], minval=c.vx_range[0], maxval=c.vx_range[1]),
        jax.random.uniform(k[1], minval=c.vy_range[0], maxval=c.vy_range[1]),
        jax.random.uniform(k[2], minval=c.wz_range[0], maxval=c.wz_range[1]),
    ])
    keep = jax.random.uniform(k[3], (3,)) >= p_zero_axis
    v = v * keep
    xy_small = jnp.linalg.norm(v[:2]) < zero_below
    return v.at[:2].set(jnp.where(xy_small, 0.0, v[:2]))


def sample_modes(rng, stage, modes_cfg, ctrl_dt):
    """(mode1, mode2, switch_step): segment modes with legal transitions."""
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    p = jnp.asarray(stage.mode_probs)
    mode1 = jax.random.choice(k1, 3, p=p / p.sum())
    # from STAND: go to LOCO or MANIP in proportion to their stage probability
    # (if both are 0, stay); from LOCO/MANIP: back to STAND.
    p_lm = p[1:]
    tot = p_lm.sum()
    to = jnp.where(tot > 0, jax.random.choice(k2, 2, p=jnp.where(tot > 0, p_lm / jnp.maximum(tot, 1e-6), jnp.array([0.5, 0.5]))) + 1, STAND)
    mode2_if = jnp.where(mode1 == STAND, to, STAND)
    switch = stage.switch_enabled & (jax.random.uniform(k3) < modes_cfg.p_switch)
    mode2 = jnp.where(switch, mode2_if, mode1)
    lo, hi = modes_cfg.switch_s
    sw = jax.random.randint(k4, (), int(lo / ctrl_dt), int(hi / ctrl_dt))
    return mode1.astype(jnp.int32), mode2.astype(jnp.int32), jnp.where(switch, sw, BIG).astype(jnp.int32)


def mode_at(info, step):
    return jnp.where(step < info["switch_step"], info["mode1"], info["mode2"])


def init_upper(rng, bank, ub_cfg, mode, arm_scale, ctrl_dt):
    """Upper-body schedule state at reset (MANIP starts from default and
    moves to its first waypoint; other modes hold default)."""
    del rng, ub_cfg, arm_scale, ctrl_dt
    return {"ub_a": bank.default, "ub_b": bank.default, "ub_t0": jnp.int32(0),
            "ub_move": jnp.int32(1), "ub_next": jnp.where(mode == MANIP, 0, BIG).astype(jnp.int32),
            "ub_target": bank.default}


def advance_upper(info, step, rng, bank, ub_cfg, arm_scale, ctrl_dt):
    """Move the upper-body target one control step along its schedule."""
    mode = mode_at(info, step)
    prev = mode_at(info, jnp.maximum(step - 1, 0))
    left_manip = (prev == MANIP) & (mode != MANIP)
    entered_manip = (prev != MANIP) & (mode == MANIP)
    start = (step >= info["ub_next"]) | left_manip | entered_manip
    k_pose, k_hold = jax.random.split(rng)
    new_b = jnp.where(mode == MANIP, sample_pose(k_pose, bank, ub_cfg, arm_scale), bank.default)
    dist = jnp.max(jnp.abs(new_b - info["ub_target"]))
    move_s = jnp.where(mode == MANIP,
                       jnp.maximum(ub_cfg.move_s_min, dist / ub_cfg.joint_speed_max),
                       ub_cfg.return_to_default_s)
    move = jnp.maximum(jnp.round(move_s / ctrl_dt).astype(jnp.int32), 1)
    hold = jax.random.randint(k_hold, (), int(ub_cfg.hold_s[0] / ctrl_dt), int(ub_cfg.hold_s[1] / ctrl_dt))
    nxt = jnp.where(mode == MANIP, step + move + hold, BIG)
    info = dict(info)
    info["ub_a"] = jnp.where(start, info["ub_target"], info["ub_a"])
    info["ub_b"] = jnp.where(start, new_b, info["ub_b"])
    info["ub_t0"] = jnp.where(start, step, info["ub_t0"])
    info["ub_move"] = jnp.where(start, move, info["ub_move"])
    info["ub_next"] = jnp.where(start, nxt, info["ub_next"])
    s = min_jerk((step - info["ub_t0"]) / info["ub_move"])
    info["ub_target"] = info["ub_a"] + s * (info["ub_b"] - info["ub_a"])
    return info


def advance_cmd(info, step, rng, c, cmd_cfg, ctrl_dt):
    """Velocity command: resampled every resample_s while in LOCO, 0 otherwise."""
    mode = mode_at(info, step)
    k_cmd, k_t = jax.random.split(rng)
    due = step >= info["cmd_next"]
    new = sample_cmd_vel(k_cmd, c, cmd_cfg.zero_below, cmd_cfg.p_zero_axis)
    lo, hi = cmd_cfg.resample_s
    dt = jax.random.randint(k_t, (), int(lo / ctrl_dt), int(hi / ctrl_dt))
    info = dict(info)
    info["cmd_raw"] = jnp.where(due, new, info["cmd_raw"])
    info["cmd_next"] = jnp.where(due, step + dt, info["cmd_next"])
    info["cmd"] = jnp.where(mode == LOCO, info["cmd_raw"], jnp.zeros(3))
    return info


def gait_clock(step, ctrl_dt, gait_cfg):
    """unitree_rl_gym G1 clock: phase = (t mod period)/period, t = step*dt from
    the start of the episode (controller: from policy activation). Returns
    (obs (2,) [sin 2pi phase, cos 2pi phase], stance (2,) bool L/R)."""
    phase = jnp.mod(step * ctrl_dt, gait_cfg.period_s) / gait_cfg.period_s
    leg = jnp.stack([phase, jnp.mod(phase + gait_cfg.offset, 1.0)])
    stance = leg < gait_cfg.stance_fraction
    obs = jnp.stack([jnp.sin(2 * jnp.pi * phase), jnp.cos(2 * jnp.pi * phase)])
    return obs, stance
