"""Teacher actor-critic (brax PPO factory) and the ULTRA-style CVAE student.

Teacher: ULTRA rlg g1_teacher.yaml — separate actor and critic MLPs
[1024, 1024, 512] relu on the full teacher observation ([state, privileged]).
brax's NormalTanhDistribution learns its std; we initialise it per joint so
the initial joint-space exploration noise equals ULTRA's
(exp(-2.9) x action_scale 3 = 0.165 rad), see train.yaml networks.

Student (ULTRA ultra_network_builder_obj_v2.py, reduced to proprioception):
  prior    p(z | h)       = N(mu_p, var_p), MLP over the flattened history h
  encoder  q(z | h, priv) = N(mu_p + mu_e, var_e)  (residual on the prior mean,
                            ULTRA kl_loss), MLP over [h, clean state, privileged]
  decoder  a = MLP([s_t, z])
Deployment: z = mu_p (ULTRA docs/deploy.md: latent noise 0), so the exported
graph is prior + decoder only: history -> action.
"""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from brax.training import distribution, networks, types
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen

ACTIVATIONS = {"relu": linen.relu, "swish": linen.swish, "elu": linen.elu, "tanh": linen.tanh}


class MLP(linen.Module):
    layer_sizes: Sequence[int]
    activation: Callable = linen.relu
    activate_final: bool = False
    final_bias: np.ndarray | None = None      # optional init of the last bias
    final_kernel_scale: float = 1.0           # <1: small initial outputs

    @linen.compact
    def __call__(self, x):
        n = len(self.layer_sizes)
        for i, size in enumerate(self.layer_sizes):
            last = i == n - 1
            bias_init = linen.initializers.zeros_init()
            if last and self.final_bias is not None:
                fb = jnp.asarray(self.final_bias)
                bias_init = lambda *_: fb  # noqa: E731
            kernel_init = linen.initializers.lecun_normal()
            if last and self.final_kernel_scale != 1.0:
                kernel_init = linen.initializers.variance_scaling(
                    self.final_kernel_scale ** 2, "fan_in", "truncated_normal")
            x = linen.Dense(size, name=f"hidden_{i}", bias_init=bias_init,
                            kernel_init=kernel_init)(x)
            if not last or self.activate_final:
                x = self.activation(x)
        return x


def _inv_softplus(y):
    return np.log(np.expm1(y))


def init_std_bias(action_scale, noise_rad, action_size, min_std=0.001):
    """Final-layer bias so brax's tanh-normal starts with per-joint std
    noise_rad / action_scale (clipped to [0.05, 1.0]); loc bias 0."""
    std = np.clip(noise_rad / np.asarray(action_scale), 0.05, 1.0)
    return np.concatenate([np.zeros(action_size), _inv_softplus(std - min_std)]).astype(np.float32)


def make_teacher_networks(observation_size, action_size,
                          preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
                          *, policy_hidden=(1024, 1024, 512), value_hidden=(1024, 1024, 512),
                          activation="relu", action_scale=None, init_joint_noise_rad=0.165):
    def _dim(x):
        return int(x[-1]) if isinstance(x, (tuple, list)) else int(x)
    s_dim = _dim(observation_size["state"])
    p_dim = _dim(observation_size["privileged_state"])
    act = ACTIVATIONS[activation]
    dist = distribution.NormalTanhDistribution(event_size=action_size)
    fb = None if action_scale is None else init_std_bias(action_scale, init_joint_noise_rad, action_size)
    # final kernel x0.01: the initial mean action is ~0, i.e. q* ~ default_qpos.
    # Without it a random 1024-wide MLP outputs O(1) means, which with the
    # 3 rad arm action scale throws the arms around from step 0.
    policy = MLP(tuple(policy_hidden) + (dist.param_size,), act, final_bias=fb,
                 final_kernel_scale=0.01)
    value = MLP(tuple(value_hidden) + (1,), act)

    def _x(obs, pp):
        obs = preprocess_observations_fn(obs, pp)
        return jnp.concatenate([obs["state"], obs["privileged_state"]], axis=-1)

    pol_net = networks.FeedForwardNetwork(
        init=lambda key: policy.init(key, jnp.zeros((1, s_dim + p_dim))),
        apply=lambda pp, params, obs: policy.apply(params, _x(obs, pp)))
    val_net = networks.FeedForwardNetwork(
        init=lambda key: value.init(key, jnp.zeros((1, s_dim + p_dim))),
        apply=lambda pp, params, obs: jnp.squeeze(value.apply(params, _x(obs, pp)), -1))
    return ppo_networks.PPONetworks(policy_network=pol_net, value_network=val_net,
                                    parametric_action_distribution=dist)


def make_networks_factory(cfg):
    import functools
    n = cfg.train.networks
    return functools.partial(make_teacher_networks, policy_hidden=tuple(n.policy_hidden),
                             value_hidden=tuple(n.value_hidden), activation=n.activation,
                             action_scale=list(cfg.actuators.action_scale),
                             init_joint_noise_rad=float(n.init_joint_noise_rad))


def teacher_policy_module(cfg, action_size):
    n = cfg.train.networks
    dist = distribution.NormalTanhDistribution(event_size=action_size)
    return MLP(tuple(n.policy_hidden) + (dist.param_size,), ACTIVATIONS[n.activation])


def deterministic_action(logits):
    """Mode of the tanh-normal = deployment action in [-1, 1]."""
    return jnp.tanh(logits[..., : logits.shape[-1] // 2])


# ------------------------------------------------------------------ student
class CVAEStudent(linen.Module):
    """ULTRA-style conditional VAE student over a proprioceptive history."""

    action_size: int
    vae_dim: int
    prior_hidden: Sequence[int]
    encoder_hidden: Sequence[int]
    decoder_hidden: Sequence[int]
    activation: Callable = linen.relu

    def setup(self):
        a = self.activation
        self.prior = MLP(tuple(self.prior_hidden) + (2 * self.vae_dim,), a, name="prior")
        self.encoder = MLP(tuple(self.encoder_hidden) + (2 * self.vae_dim,), a, name="encoder")
        self.decoder = MLP(tuple(self.decoder_hidden) + (self.action_size,), a, name="decoder")

    def prior_dist(self, hist):
        out = self.prior(hist.reshape(hist.shape[:-2] + (-1,)))
        mu, logvar = jnp.split(out, 2, axis=-1)
        return mu, jnp.clip(logvar, -10.0, 4.0)

    def encoder_dist(self, hist, teacher_x):
        x = jnp.concatenate([hist.reshape(hist.shape[:-2] + (-1,)), teacher_x], axis=-1)
        mu, logvar = jnp.split(self.encoder(x), 2, axis=-1)
        return mu, jnp.clip(logvar, -10.0, 4.0)

    def decode(self, s, z):
        return jnp.tanh(self.decoder(jnp.concatenate([s, z], axis=-1)))

    def __call__(self, hist, teacher_x, eps):
        """Training pass: returns (action_posterior, action_prior_mean,
        (mu_p, logvar_p), (mu_e, logvar_e))."""
        mu_p, lv_p = self.prior_dist(hist)
        mu_e, lv_e = self.encoder_dist(hist, teacher_x)
        z = mu_p + mu_e + jnp.exp(0.5 * lv_e) * eps
        s = hist[..., -1, :]
        return self.decode(s, z), self.decode(s, mu_p), (mu_p, lv_p), (mu_e, lv_e)

    def act(self, hist):
        """Deployment: z = prior mean."""
        mu_p, _ = self.prior_dist(hist)
        return self.decode(hist[..., -1, :], mu_p)


def make_student(cfg, action_size):
    s = cfg.train.student
    return CVAEStudent(action_size=action_size, vae_dim=int(s.vae_dim),
                       prior_hidden=tuple(s.prior_hidden), encoder_hidden=tuple(s.encoder_hidden),
                       decoder_hidden=tuple(s.decoder_hidden), activation=ACTIVATIONS[s.activation])


def kl_residual(mu_p, lv_p, mu_e, lv_e):
    """ULTRA kl_loss: KL(N(mu_p + mu_e, var_e) || N(mu_p, var_p)), per dim."""
    return 0.5 * (lv_p - lv_e + jnp.exp(lv_e) / jnp.exp(lv_p) + mu_e ** 2 / jnp.exp(lv_p) - 1.0)
