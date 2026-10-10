"""Curriculum and config invariants (skill L13, M8, M2)."""
import numpy as np
import pytest

from g1_wbc import curriculum
from g1_wbc.config import load_config, stage_domain_rand

cfg = load_config()
FACTORS = ("cmd_scale", "dr_scale", "push_scale", "arm_scale", "switch_enabled")


@pytest.mark.parametrize("cur", ["stand", "teleop"])
def test_one_factor_per_stage(cur):
    st = cfg.stages(cur)
    for a, b in zip(st, st[1:]):
        changed = [f for f in FACTORS if getattr(a, f) != getattr(b, f)]
        assert len(changed) <= 1, f"{a.name}->{b.name} changes {changed} (L13)"


def test_final_stages_full_spec():
    a = cfg.stages("stand")[-1]
    assert (a.dr_scale, a.push_scale) == (1.0, 1.0)
    b = cfg.stages("teleop")[-1]
    assert (b.cmd_scale, b.dr_scale, b.push_scale, b.arm_scale) == (1.0, 1.0, 1.0, 1.0)
    assert b.switch_enabled and all(p > 0 for p in b.mode_probs)


def test_tracking_gates_are_floor_midpoints():
    for s in cfg.stages("teleop"):
        f = curriculum.do_nothing_floor(cfg, s)
        assert abs(s.gate_tracking - (f + 1) / 2) < 0.006, (s.name, f)


def test_command_envelope_contains_contract_limits():
    c = cfg.commands.cmd_vel   # docs/interfaces.md 3.3
    assert c.vx_range == [-0.5, 0.8] and c.vy_range == [-0.4, 0.4] and c.wz_range == [-1.0, 1.0]


def test_dr_scale_zero_is_nominal_and_one_is_ultra():
    st = cfg.stages("stand")
    d0, d1 = stage_domain_rand(cfg.domain_rand, st[0]), stage_domain_rand(cfg.domain_rand, st[-1])
    assert d0.ground.friction_range == [1.0, 1.0] and d0.actuation.action_delay_steps == [0, 0]
    assert d1.ground.friction_range == [0.5, 2.0]            # ULTRA friction_range
    assert d1.actuation.motor_strength_range == [0.8, 1.2]   # ULTRA motor_strength_range
    assert d1.push.max_vel_xy == 1.0                         # ULTRA max_push_vel_xy


def test_ppo_geometry():
    p = cfg.train.ppo
    assert (p.batch_size * p.num_minibatches) % p.num_envs == 0
    assert p.batch_size * p.unroll_length == 10240           # H1-2 minibatch (L12)


def test_effort_switch():
    from g1_wbc.layout import effort
    assert cfg.actuators.effort_source in ("canonical", "ultra")
    assert effort(cfg.actuators)[4] == (35.0 if cfg.actuators.effort_source == "canonical" else 50.0)
