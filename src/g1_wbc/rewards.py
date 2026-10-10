"""Stage A/B reward terms. Pure functions of plain arrays (unit-testable with
hand-built states); every weight/width in configs/rewards.yaml. Sources are
cited per term in rewards.yaml (URG = unitree_rl_gym G1, ULTRA smooth =
ULTRA smooth_rewards.py, SKILL = H1-2/X1-validated)."""

from typing import NamedTuple

import jax.numpy as jnp


class RewardInputs(NamedTuple):
    cmd: jnp.ndarray              # (3,) vx, vy, wz
    lin_vel_local: jnp.ndarray    # (3,)
    ang_vel_local: jnp.ndarray    # (3,)
    proj_gravity: jnp.ndarray     # (3,)
    base_height: jnp.ndarray      # ()
    base_acc: jnp.ndarray         # (3,) world
    upper_q: jnp.ndarray          # (17,) waist + arm joint positions
    upper_target: jnp.ndarray     # (17,) commanded
    q: jnp.ndarray                # (29,)
    qd: jnp.ndarray               # (29,)
    q_lo: jnp.ndarray             # (29,) joint limits
    q_hi: jnp.ndarray
    tau: jnp.ndarray              # (29,) last substep torque
    tau_limit: jnp.ndarray        # (29,)
    action: jnp.ndarray           # (29,)
    prev_action: jnp.ndarray
    dqd_sub: jnp.ndarray          # () sum over substeps of |qd_k - qd_0|^2 (ULTRA delta_dof_vel_all)
    dw_sub: jnp.ndarray           # () same for pelvis angular velocity
    feet_contact: jnp.ndarray     # (2,) bool
    first_contact: jnp.ndarray    # (2,) bool
    feet_air_time: jnp.ndarray    # (2,) s
    feet_height: jnp.ndarray      # (2,) sole height above floor
    feet_vel: jnp.ndarray         # (2, 3) world
    feet_gravity_xy: jnp.ndarray  # (2, 2) gravity in foot frame, x/y
    feet_lateral: jnp.ndarray     # () foot-foot horizontal distance
    knee_lateral: jnp.ndarray     # () knee-knee horizontal distance
    stance_clock: jnp.ndarray     # (2,) bool: gait clock says foot in stance
    hip_roll_yaw: jnp.ndarray     # (4,)


def is_standing(cmd, threshold):
    return jnp.linalg.norm(cmd) < threshold


def tracking_lin_vel(cmd, v, sigma):
    return jnp.exp(-jnp.sum(jnp.square(cmd[:2] - v[:2])) / sigma)


def tracking_yaw(cmd, w, sigma):
    return jnp.exp(-jnp.square(cmd[2] - w[2]) / sigma)


def tracking_upper(q, target, sigma):
    return jnp.exp(-jnp.mean(jnp.square(q - target)) / sigma)


def tilt_exceeded(proj_g, max_tilt, max_pitch_forward, xp=jnp):
    """Shared by env (jnp) and CPU evaluator (numpy); they MUST agree (M1).
    Ellipse in (forward pitch, roll) stretched forward by
    max_pitch_forward / max_tilt; roll and backward pitch keep max_tilt."""
    s, fwd = xp.sin(max_tilt), xp.sin(max_pitch_forward)
    px = xp.where(proj_g[0] > 0.0, proj_g[0] * (s / fwd), proj_g[0])
    past_horizontal = -proj_g[2] < 0.0
    return (xp.sqrt(px * px + proj_g[1] * proj_g[1]) > s) | past_horizontal


def band_violation(d, band):
    """ULTRA smooth_rewards feet/knee_distance: (|clamp(d-lo,-0.5,0)|*100 +
    |clamp(d-hi,0,1)|*100) / 2."""
    lo, hi = band
    return (jnp.abs(jnp.clip(d - lo, -0.5, 0.0)) * 100.0
            + jnp.abs(jnp.clip(d - hi, 0.0, 1.0)) * 100.0) / 2.0


def gait_contact(contact, stance_clock):
    """URG _reward_contact: +1 per foot whose contact matches the clock."""
    return jnp.sum(~(contact ^ stance_clock)).astype(jnp.float32)


def dof_pos_limits(q, lo, hi, soft):
    """URG soft limits: the middle `soft` fraction of the range is free."""
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * soft
    return jnp.sum(jnp.clip((mid - half) - q, 0.0) + jnp.clip(q - (mid + half), 0.0))


def compute_reward(x: RewardInputs, cfg, height_target) -> tuple:
    """(total, dict of RAW term values). Stand/manip (|cmd| small) switches the
    gait terms off and the both-feet stance bonus on."""
    w, k = cfg.weights, cfg.kernels
    standing = is_standing(x.cmd, cfg.standing_cmd_threshold)
    walking = 1.0 - standing.astype(jnp.float32)
    contact_f = x.feet_contact.astype(jnp.float32)
    terms = {
        "tracking_lin_vel": tracking_lin_vel(x.cmd, x.lin_vel_local, k.tracking_sigma),
        "tracking_yaw": tracking_yaw(x.cmd, x.ang_vel_local, k.tracking_yaw_sigma),
        "tracking_upper": tracking_upper(x.upper_q, x.upper_target, k.upper_sigma),
        "base_height": jnp.square(x.base_height - height_target),
        "alive": jnp.float32(1.0),
        "orientation": jnp.sum(jnp.square(x.proj_gravity[:2])),
        "torso_upright": jnp.exp(-jnp.sum(jnp.square(x.proj_gravity[:2])) / k.upright_sigma),
        "lin_vel_z": jnp.square(x.lin_vel_local[2]),
        "ang_vel_xy": jnp.sum(jnp.square(x.ang_vel_local[:2])),
        "base_acc": jnp.sum(jnp.square(x.base_acc)),
        "contact": walking * gait_contact(x.feet_contact, x.stance_clock),
        "feet_swing_height": walking * jnp.sum(
            jnp.square(x.feet_height - cfg.swing_height_target) * (1.0 - contact_f)),
        "contact_no_vel": jnp.sum(jnp.square(x.feet_vel) * contact_f[:, None]),
        "hip_pos": jnp.sum(jnp.square(x.hip_roll_yaw)),
        "feet_air_time": walking * jnp.sum(x.feet_air_time * x.first_contact),
        "feet_stance": (standing & jnp.all(x.feet_contact)).astype(jnp.float32),
        "action_rate": jnp.linalg.norm(x.action - x.prev_action),
        "torques": jnp.linalg.norm(x.tau),
        "dof_vel": jnp.sum(jnp.square(x.qd)),
        "dof_pos_limits": dof_pos_limits(x.q, x.q_lo, x.q_hi, cfg.soft_dof_pos_limit),
        "dof_torque_limits": jnp.sum(jnp.clip(jnp.abs(x.tau) / x.tau_limit
                                              - cfg.soft_torque_limit, 0.0)),
        "energy": jnp.linalg.norm(jnp.abs(x.tau * x.qd)),
        "delta_dof_vel_all": x.dqd_sub,
        "delta_ang_vel_all": x.dw_sub,
        "feet_orientation": jnp.sum(jnp.linalg.norm(x.feet_gravity_xy, axis=-1)),
        "foot_slip": jnp.sum(jnp.sqrt(jnp.linalg.norm(x.feet_vel[:, :2], axis=-1) + 1e-8)
                             * contact_f),
        "feet_distance": band_violation(x.feet_lateral, cfg.feet_distance_band),
        "knee_distance": band_violation(x.knee_lateral, cfg.knee_distance_band),
    }
    missing = set(terms) - set(w) | (set(w) - set(terms) - {"termination"})
    if missing:
        raise KeyError(f"reward terms vs weights mismatch: {sorted(missing)}")
    total = sum(w[n] * v for n, v in terms.items())
    return total, terms
