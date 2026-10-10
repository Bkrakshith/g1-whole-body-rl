"""Curriculum gates (skill theory 7, failures M8, L4, L5, L16).

brax logs env metrics as eval/episode_<name> summed over the episode; per-step
values are that / eval/avg_episode_length."""

import numpy as np

from .config import Stage, stage_commands

EPISODE_STEPS = 1000     # set by ppo_teacher from train.ppo.episode_length


def _per_step(m, key):
    return float(m.get(key, 0.0)) / (float(m.get("eval/avg_episode_length", 1.0)) or 1.0)


def survival_fraction(m):
    return float(m.get("eval/avg_episode_length", 0.0)) / float(EPISODE_STEPS)


def gate_metrics(m):
    return {
        "tracking": _per_step(m, "eval/episode_reward/tracking_lin_vel"),
        "air": _per_step(m, "eval/episode_reward/feet_air_time"),
        "upper": _per_step(m, "eval/episode_reward/tracking_upper"),
        "survival": survival_fraction(m),
    }


def gate_passed(stage: Stage, m) -> tuple[bool, dict]:
    g = gate_metrics(m)
    ok = (g["tracking"] >= stage.gate_tracking and g["air"] >= stage.gate_air_time
          and g["upper"] >= stage.gate_upper and g["survival"] >= stage.gate_survival)
    return ok, g


def stall_report(stage: Stage, g: dict) -> str:
    return (
        f"\n{'=' * 72}\nCURRICULUM STALL: stage '{stage.name}' never passed its gates.\n"
        f"  tracking {g['tracking']:.3f}/{stage.gate_tracking}  air {g['air']:.4f}/"
        f"{stage.gate_air_time}  upper {g['upper']:.3f}/{stage.gate_upper}  survival "
        f"{g['survival']:.3f}/{stage.gate_survival}\n"
        "Per the skill (5.4): do NOT shrink the spec. In order:\n"
        "  1. more budget for this stage (a young gait is being refined)\n"
        "  2. one-factor ablation of the last passing checkpoint (failure-modes.md)\n"
        "     to find which new factor blocks it; give it its own stage\n"
        "  3. check the normaliser (M12), termination headroom (L1), kernel widths (L3/L17)\n"
        f"{'=' * 72}\n")


def do_nothing_floor(cfg, stage: Stage, n=200_000, seed=0) -> float:
    """Per-step lin-vel tracking kernel a policy that never moves would earn
    (stand/manip segments pay exp(0) = 1 for free). Monte Carlo over the SAME
    command distribution commands.sample_cmd_vel / sample_modes draw from
    (approximating the in-episode command resampling by i.i.d. draws, exact in
    expectation). Gates are set at (floor + 1) / 2 (tests pin it)."""
    rng = np.random.default_rng(seed)
    c = stage_commands(cfg.commands, stage)
    cv = cfg.commands.cmd_vel
    sigma = cfg.rewards.kernels.tracking_sigma
    v = np.stack([rng.uniform(*c.vx_range, n), rng.uniform(*c.vy_range, n)], -1)
    keep = rng.uniform(size=(n, 2)) >= cv.p_zero_axis
    v = v * keep
    v[np.linalg.norm(v, axis=-1) < cv.zero_below] = 0.0
    walk_kernel = np.exp(-np.sum(v ** 2, -1) / sigma).mean()
    p = np.asarray(stage.mode_probs, float)
    p = p / p.sum()
    p_loco = p[1]
    if stage.switch_enabled:
        # a switch replaces a fraction of the episode by the other mode; loco
        # segments switch to stand, stand segments to loco with p[1]/(p1+p2)
        ep = cfg.train.env.episode_length_s
        lo, hi = cfg.commands.modes.switch_s
        frac_after = 1.0 - (lo + hi) / 2 / ep
        ps = cfg.commands.modes.p_switch
        to_loco = p[1] / max(p[1] + p[2], 1e-9)
        p_loco = (p[1] * (1 - ps * frac_after) + p[0] * ps * frac_after * to_loco)
    return float(1.0 - p_loco + p_loco * walk_kernel)
