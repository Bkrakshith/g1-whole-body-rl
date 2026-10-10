"""Config loading. All physical ranges, reward weights and curriculum
thresholds live in configs/*.yaml; this module only parses them."""

import dataclasses
import json
import os
from typing import Any

import yaml

RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(RL_ROOT, "configs")
ASSETS_DIR = os.path.join(RL_ROOT, "assets")
MJX_XML = os.path.join(ASSETS_DIR, "g1_mjx.xml")
EVAL_XML = os.path.join(ASSETS_DIR, "g1_eval.xml")
MODEL_INFO = os.path.join(ASSETS_DIR, "model_info.json")


class DotDict(dict):
    """dict with attribute access, recursively applied."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj


@dataclasses.dataclass(frozen=True)
class Stage:
    """One curriculum stage. Frozen: a stage change = new env + fresh jit."""

    name: str
    num_timesteps: int
    cmd_scale: float          # velocity command ranges x this
    dr_scale: float           # every DR range pulled toward nominal by this
    push_scale: float         # push velocity x this
    arm_scale: float          # upper-body targets blended default -> full by this
    mode_probs: tuple         # (stand, loco, manip) per segment
    switch_enabled: bool      # second segment with a mode switch
    gate_tracking: float = 0.0
    gate_air_time: float = 0.0
    gate_survival: float = 0.0
    gate_upper: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "mode_probs", tuple(float(p) for p in self.mode_probs))


def _load_yaml(name: str) -> DotDict:
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return DotDict.wrap(yaml.safe_load(f))


def model_info() -> DotDict:
    with open(MODEL_INFO) as f:
        return DotDict.wrap(json.load(f))


def stage_commands(commands, stage: Stage):
    """Velocity ranges of a stage (x cmd_scale)."""
    k = float(stage.cmd_scale)
    c = dict(commands.cmd_vel)
    for r in ("vx_range", "vy_range", "wz_range"):
        c[r] = [k * float(v) for v in commands.cmd_vel[r]]
    return DotDict.wrap(c)


# Nominal value each DR range collapses to at dr_scale = 0.
_DR_NOMINAL = {
    ("ground", "friction_range"): 1.0,
    ("body", "torso_mass_delta_range"): 0.0,
    ("body", "torso_com_shift_range"): 0.0,
    ("actuation", "motor_strength_range"): 1.0,
    ("gravity", "offset_range"): 0.0,
}


def stage_domain_rand(dr, stage: Stage):
    """DR of a stage: ranges pulled to nominal by dr_scale, the action delay
    upper bound rounded, obs noise scaled; pushes scaled by push_scale."""
    k = float(stage.dr_scale)
    out = {sec: dict(dr[sec]) for sec in ("ground", "body", "actuation", "gravity",
                                            "obs_noise", "push")}
    for (sec, key), nom in _DR_NOMINAL.items():
        out[sec][key] = [nom + k * (float(v) - nom) for v in dr[sec][key]]
    lo, hi = dr.actuation.action_delay_steps
    out["actuation"]["action_delay_steps"] = [int(round(k * lo)), int(round(k * hi))]
    out["obs_noise"] = {n: (k * float(v) if n != "distribution" else v)
                        for n, v in dr.obs_noise.items()}
    out["push"]["max_vel_xy"] = float(stage.push_scale) * float(dr.push.max_vel_xy)
    return DotDict.wrap({**dr, **out})


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = _deep_merge(base[k], v) if k in base else v
        return DotDict.wrap(out)
    return override


@dataclasses.dataclass
class Config:
    actuators: DotDict
    rewards: DotDict
    domain_rand: DotDict
    commands: DotDict
    train: DotDict

    def stages(self, curriculum: str = "teleop") -> list[Stage]:
        return [Stage(**s) for s in self.train.curricula[curriculum]]


def load_config(overlay: str | None = None) -> Config:
    """Base config, optionally with an overlay (configs/<overlay>) holding only
    deltas; its top-level keys must be Config field names."""
    cfg = Config(
        actuators=_load_yaml("actuators.yaml"),
        rewards=_load_yaml("rewards.yaml"),
        domain_rand=_load_yaml("domain_rand.yaml"),
        commands=_load_yaml("commands.yaml"),
        train=_load_yaml("train.yaml"),
    )
    if overlay is None:
        return cfg
    extra = _load_yaml(overlay)
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(extra) - known
    if unknown:
        raise ValueError(f"{overlay}: unknown sections {sorted(unknown)}")
    for section, value in extra.items():
        setattr(cfg, section, _deep_merge(getattr(cfg, section), value))
    return cfg


def base_height_target(cfg: Config) -> float:
    t = cfg.rewards.get("base_height_target")
    return float(t) if t is not None else float(model_info().train.info.pelvis_height_home_m)
