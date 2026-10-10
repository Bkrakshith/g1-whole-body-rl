"""Per-episode domain randomisation (ULTRA ranges, configs/domain_rand.yaml).

Every field is sampled at every reset and carried in state.info; the env
rebuilds the mjx.Model view per env each step (model.replace), so auto-reset
really re-randomises (unlike brax's per-instance randomization_fn)."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .layout import NU


class ModelBase(NamedTuple):
    body_mass: jax.Array          # (nbody,)
    body_ipos: jax.Array          # (nbody, 3)
    geom_friction: jax.Array      # (ngeom, 3)
    torso_id: int
    friction_geom_ids: jax.Array  # floor + every sole sphere


def _u(rng, lo, hi, shape=()):
    return jax.random.uniform(rng, shape, minval=lo, maxval=hi)


def sample_model_fields(rng, base: ModelBase, dr) -> tuple[dict, dict]:
    """Returns (model field overrides, privileged latents)."""
    k = jax.random.split(rng, 7)
    friction = _u(k[0], *dr.ground.friction_range)
    geom_friction = base.geom_friction.at[base.friction_geom_ids, 0].set(friction)
    dm = _u(k[1], *dr.body.torso_mass_delta_range)
    body_mass = base.body_mass.at[base.torso_id].add(dm)
    dc = _u(k[2], *dr.body.torso_com_shift_range, (3,))
    body_ipos = base.body_ipos.at[base.torso_id].add(dc)
    lo, hi = dr.actuation.motor_strength_range
    kp_scale = _u(k[3], lo, hi, (NU,))       # ULTRA motor_strength[0], per DoF
    kd_scale = _u(k[4], lo, hi, (NU,))       # ULTRA motor_strength[1], per DoF
    d_lo, d_hi = dr.actuation.action_delay_steps
    delay = jax.random.randint(k[5], (), d_lo, d_hi + 1)
    grav = _u(k[6], *dr.gravity.offset_range, (3,))
    fields = {"body_mass": body_mass, "body_ipos": body_ipos,
              "geom_friction": geom_friction}
    priv = {"friction": friction, "torso_mass_delta": dm, "torso_com_shift": dc,
            "kp_scale": kp_scale, "kd_scale": kd_scale, "action_delay": delay,
            "gravity_offset": grav}
    return fields, priv


def sample_gravity(rng, dr):
    return _u(rng, *dr.gravity.offset_range, (3,))


def sample_push_velocity(rng, dr):
    """ULTRA humanoid.py _push_robots: base lin vel x/y SET to U(-v, v)."""
    v = dr.push.max_vel_xy
    return _u(rng, -v, v, (2,))


def sample_uniform_noise(rng, amplitude):
    """ULTRA noise: (2 * rand - 1) * amplitude (uniform, NOT Gaussian; M13)."""
    return (2.0 * jax.random.uniform(rng, amplitude.shape) - 1.0) * amplitude
