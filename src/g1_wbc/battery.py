"""Evaluation battery (skill Phase 6): scenarios derived from the configs,
all inside the training envelope (M2), >= 50 episodes each for claims,
Wilson CIs. Stage A ("stand") uses the stand scenarios only."""

import os

import numpy as np
import yaml

from .config import RL_ROOT
from .sim2sim import Scenario


def _poses():
    with open(os.path.join(RL_ROOT, "configs", "upper_body_poses.yaml")) as f:
        return np.array([p["q"] for p in yaml.safe_load(f)["library"]])


def _kicks(rng, v, t0=2.0, every=4.0, dur=20.0):
    out, t = [], t0
    while t < dur - 1.0:
        ang = rng.uniform(0, 2 * np.pi)
        out.append((t, [v * np.cos(ang), v * np.sin(ang)]))
        t += every
    return out


def _manip_seq(rng, poses, default, t0, t1, speed=2.0, hold=(0.5, 2.0)):
    seq, t, cur = [], t0, default
    while t < t1:
        p = poses[rng.integers(len(poses))]
        mv = max(0.4, np.max(np.abs(p - cur)) / speed)
        seq.append((t, t + mv, p))
        t += mv + rng.uniform(*hold)
        cur = p
    seq.append((t, t + 1.0, default))       # contract: blend back over ~1 s
    return seq


def scenarios(cfg, which="teleop", seed=0, push_v=0.7):
    """push_v: kick magnitude (m/s) inside the final stage's +-1.0 m/s per axis."""
    rng = np.random.default_rng(seed)
    c = cfg.commands.cmd_vel
    q0 = np.asarray(cfg.actuators.default_pose)
    from .layout import UPPER_IDX
    default = q0[list(UPPER_IDX)]
    poses = _poses()
    vx, vy, wz = c.vx_range, c.vy_range, c.wz_range
    s = [Scenario("stand", 20.0),
         Scenario("stand_kicks", 20.0, kicks=_kicks(rng, push_v))]
    if which == "stand":
        return s
    s += [
        Scenario("walk_fwd", 20.0, cmd=[(1.0, [0.6 * vx[1], 0, 0])]),
        Scenario("walk_fwd_max", 20.0, cmd=[(1.0, [vx[1], 0, 0])]),
        Scenario("walk_back", 20.0, cmd=[(1.0, [vx[0], 0, 0])]),
        Scenario("strafe", 20.0, cmd=[(1.0, [0, 0.75 * vy[1], 0]), (10.0, [0, 0.75 * vy[0], 0])]),
        Scenario("turn", 20.0, cmd=[(1.0, [0, 0, 0.8 * wz[1]]), (10.0, [0, 0, 0.8 * wz[0]])]),
        Scenario("stop_and_go", 20.0, cmd=[(1.0, [0.5, 0, 0]), (5.0, [0, 0, 0]), (8.0, [-0.3, 0, 0]),
                                           (11.0, [0, 0, 0]), (14.0, [0.3, 0.2, 0.5]), (18.0, [0, 0, 0])]),
        Scenario("manip", 20.0, upper=_manip_seq(rng, poses, default, 1.0, 17.0)),
        Scenario("manip_then_walk", 20.0, upper=_manip_seq(rng, poses, default, 1.0, 8.0),
                 cmd=[(12.0, [0.5, 0, 0.3])]),
        Scenario("walk_kicks", 20.0, cmd=[(1.0, [0.5, 0, 0])], kicks=_kicks(rng, push_v, t0=3.0)),
    ]
    return s
