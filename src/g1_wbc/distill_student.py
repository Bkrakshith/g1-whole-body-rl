"""Teacher -> student distillation (DAgger), ULTRA-style CVAE student.

Port of ULTRA ultra/learning/ultra_agent_distill_vae.py (the PPO term is
multiplied by 0 there, so distillation is pure imitation) onto the skill's
validated brax/JAX DAgger loop (Mujoco_agibot distill_student.py):

  rollouts : final-stage env with FULL ULTRA sensor noise on the student's
             inputs (student_noise=True) and clean state for the teacher's
             labels (emit_clean_state=True, skill L15/L21);
             each env-step acts with the teacher w.p. beta_t, else with the
             DEPLOYMENT student (prior mean); beta_t = ULTRA
             max(1 - max((it - 50) / 500, 0), 0)
  loss     : expert_coef * sum_j (a_post - a_teacher)^2           (ULTRA e_loss)
             + kld_coef(it) * KL(q(z|h,priv) || p(z|h))            (ULTRA kl_loss)
             + prior_coef(it) * sum_j (a_prior_mean - a_teacher)^2 (ULTRA prior_action_loss)
             + smooth_coef(it) * |z_mean_t - z_mean_{t-1}|^2       (ULTRA smooth_loss)
             cosine schedules as ULTRA (train.yaml distill).
Normalisation: the teacher's running 'state' statistics, frozen, applied to
every history frame (baked into the ONNX graph at export).
History: H = 10 frames oldest-first, reset-filled with the first observation
of the episode (ULTRA obs_vae.py: `obs_history_buf = stack([obs] * H)`).
"""

import functools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import Config, load_config
from .env import G1TeleopEnv
from .layout import N_ACT
from .networks import deterministic_action, kl_residual, make_student, teacher_policy_module
from .ppo_teacher import ckpt_dir, load_checkpoint, rehydrate_normalizer, save_checkpoint
from .wrappers import wrap_for_training


def _cos_ramp(it, lo, hi, it0, it1):
    p = jnp.clip((it - it0) / max(it1 - it0, 1), 0.0, 1.0)
    return lo + (hi - lo) * (1 - jnp.cos(p * jnp.pi)) / 2


def _norm(x, mean, std):
    return (x - mean) / std


def distill(teacher_ckpt, run_name, cfg: Config | None = None, num_envs=None,
            num_iterations=None, curriculum_name="teleop", env_fn=None, out_root=None):
    cfg = cfg or load_config()
    d = cfg.train.distill
    H = int(cfg.train.student.history_len)
    B = int(num_envs or d.num_envs)
    T = int(d.unroll_length)
    n_iter = int(num_iterations or d.num_iterations)

    tparams, tmeta = load_checkpoint(teacher_ckpt)
    norm = rehydrate_normalizer(tparams[0])
    s_mean, s_std = jnp.asarray(norm.mean["state"]), jnp.asarray(norm.std["state"])
    p_mean, p_std = jnp.asarray(norm.mean["privileged_state"]), jnp.asarray(norm.std["privileged_state"])
    t_pol = teacher_policy_module(cfg, N_ACT)
    t_params = tparams[1]

    stage = cfg.stages(curriculum_name)[-1]
    env = (env_fn or (lambda c: G1TeleopEnv(c, stage, student_noise=True,
                                            emit_clean_state=True)))(cfg)
    wenv = wrap_for_training(env, int(cfg.train.ppo.episode_length), 1)
    student = make_student(cfg, N_ACT)
    vae = int(cfg.train.student.vae_dim)

    key = jax.random.PRNGKey(int(d.seed))
    key, k_init, k_reset = jax.random.split(key, 3)
    obs_dim = env.observation_size["state"]
    tx_dim = obs_dim + env.observation_size["privileged_state"]
    params = student.init(k_init, jnp.zeros((1, H, obs_dim)), jnp.zeros((1, tx_dim)),
                          jnp.zeros((1, vae)))
    sched = optax.join_schedules(
        [optax.constant_schedule(float(d.learning_rate)),
         optax.linear_schedule(float(d.learning_rate), float(d.learning_rate_final),
                               max(n_iter - int(d.beta_decay_iters), 1))],
        [int(d.beta_decay_iters)])
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = tx.init(params)

    env_state = jax.jit(wenv.reset)(jax.random.split(k_reset, B))
    first = _norm(env_state.obs["state"], s_mean, s_std)
    hist = jnp.repeat(first[:, None], H, axis=1)

    def teacher_act(s_clean_n, p_n):
        return deterministic_action(t_pol.apply(t_params, jnp.concatenate([s_clean_n, p_n], -1)))

    @jax.jit
    def rollout(params, env_state, hist, beta, rng):
        def step_fn(carry, r):
            env_state, hist = carry
            o = env_state.obs
            s_n = _norm(o["state"], s_mean, s_std)
            sc_n = _norm(o["state_clean"], s_mean, s_std)
            p_n = _norm(o["privileged_state"], p_mean, p_std)
            hist = jnp.roll(hist, -1, axis=1).at[:, -1].set(s_n)
            a_t = teacher_act(sc_n, p_n)
            a_s = student.apply(params, hist, method="act")
            use_t = jax.random.uniform(r, (B, 1)) < beta
            nxt = wenv.step(env_state, jnp.where(use_t, a_t, a_s))
            done = nxt.done
            # reset-fill: the NEW episode's first observation fills the history
            new_first = _norm(nxt.obs["state"], s_mean, s_std)
            hist_next = jnp.where(done[:, None, None] > 0,
                                  jnp.repeat(new_first[:, None], H, axis=1), hist)
            # note: hist_next's last row is overwritten with the same frame next step
            return (nxt, hist_next), (hist, jnp.concatenate([sc_n, p_n], -1), a_t, done)

        (env_state, hist), batch = jax.lax.scan(step_fn, (env_state, hist), jax.random.split(rng, T))
        return env_state, hist, batch, jnp.mean(batch[3])

    def loss_fn(params, batch, it, rng):
        h, tx_in, a_t, done = batch                    # (T, b, ...)
        eps = jax.random.normal(rng, h.shape[:2] + (vae,))
        a_post, a_prior, (mu_p, lv_p), (mu_e, lv_e) = jax.vmap(
            lambda hh, xx, ee: student.apply(params, hh, xx, ee))(h, tx_in, eps)
        e_loss = jnp.mean(jnp.sum(jnp.square(a_post - a_t), -1))
        prior_loss = jnp.mean(jnp.sum(jnp.square(a_prior - a_t), -1))
        kld = jnp.mean(jnp.sum(kl_residual(mu_p, lv_p, mu_e, lv_e), -1))
        z = mu_p + mu_e
        same_ep = (1.0 - done[:-1])[..., None]
        smooth = jnp.mean(jnp.sum(jnp.square(z[1:] - z[:-1]) * same_ep, -1))
        i0, i1 = d.schedule_iters
        kc = _cos_ramp(it, *d.kld_coef, i0, i1)
        pc = _cos_ramp(it, *d.prior_action_coef, i0, i1)
        sc = _cos_ramp(it, *d.smooth_coef, i0, (i0 + i1) // 2)
        loss = float(d.expert_loss_coef) * e_loss + kc * kld + pc * prior_loss + sc * smooth
        return loss, (e_loss, prior_loss, kld)

    n_mb, n_ep = int(d.num_minibatches), int(d.num_epochs)

    @jax.jit
    def update(params, opt_state, batch, it, rng):
        mb = B // n_mb

        def epoch(carry, r):
            params, opt_state = carry
            r_perm, r_eps = jax.random.split(r)
            perm = jax.random.permutation(r_perm, B)[: mb * n_mb].reshape(n_mb, mb)

            def mb_step(carry, xs):
                params, opt_state = carry
                idx, re = xs
                b = jax.tree.map(lambda x: x[:, idx], batch)   # keep the time axis
                (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, b, it, re)
                upd, opt_state = tx.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), aux

            (params, opt_state), aux = jax.lax.scan(
                mb_step, (params, opt_state), (perm, jax.random.split(r_eps, n_mb)))
            return (params, opt_state), jax.tree.map(jnp.mean, aux)

        (params, opt_state), aux = jax.lax.scan(epoch, (params, opt_state), jax.random.split(rng, n_ep))
        return params, opt_state, jax.tree.map(lambda x: x[-1], aux)

    run_dir = os.path.join(out_root, run_name) if out_root else ckpt_dir(cfg, run_name)
    os.makedirs(run_dir, exist_ok=True)
    meta = {"phase": "distill", "teacher_ckpt": os.path.abspath(teacher_ckpt),
            "teacher_stage": tmeta.get("stage_name"), "history_len": H, "vae_dim": vae,
            "curriculum": curriculum_name, "stage": stage.name, "run_name": run_name}

    def save(name, it, prior_loss):
        save_checkpoint(os.path.join(run_dir, name),
                        {"normalizer": tparams[0], "student": params},
                        {**meta, "iteration": int(it), "prior_action_loss": float(prior_loss)})

    best, best_it, t_save = float("inf"), 0, time.time()
    hold, decay = int(d.beta_hold_iters), int(d.beta_decay_iters)
    it = 0
    for it in range(n_iter):
        beta = max(1.0 - max((it - hold) / decay, 0.0), 0.0)
        key, kr, ku = jax.random.split(key, 3)
        env_state, hist, batch, done_rate = rollout(params, env_state, hist, beta, kr)
        params, opt_state, (e_l, p_l, kld) = update(params, opt_state, batch, it, ku)
        p_l = float(p_l)
        if it % 10 == 0 or it == n_iter - 1:
            print(f"[distill] it={it} beta={beta:.2f} expert={float(e_l):.4f} "
                  f"prior_action={p_l:.4f} kl={float(kld):.3f} "
                  f"done_rate={float(done_rate) * 1e3:.2f}/1000 steps", flush=True)
        if beta == 0.0 and p_l < best - 1e-5:
            best, best_it = p_l, it
            save("ckpt_best", it, p_l)
        if time.time() - t_save > cfg.train.checkpoint.interval_s:
            save(f"ckpt_{it}", it, p_l)
            t_save = time.time()
        if beta == 0.0 and it - max(best_it, hold + decay) > int(d.plateau_patience):
            print(f"[distill] plateau (best prior-action loss {best:.4f} at it {best_it})")
            break
    save("ckpt_final", it, p_l)
    print(f"[distill] done -> {run_dir}/ckpt_best (deployment-path loss after beta=0), ckpt_final")
    return os.path.join(run_dir, "ckpt_final")


class StudentJax:
    """Deployment computation of a distilled student in JAX: raw history
    (H, obs) -> action. Exactly what the ONNX export freezes."""

    def __init__(self, ckpt, cfg=None):
        cfg = cfg or load_config()
        p, self.meta = load_checkpoint(ckpt)
        norm = rehydrate_normalizer(p["normalizer"])
        self.mean = np.asarray(norm.mean["state"])
        self.std = np.asarray(norm.std["state"])
        self.params = p["student"]
        self.module = make_student(cfg, N_ACT)
        mean, std = jnp.asarray(self.mean), jnp.asarray(self.std)
        mod, prm = self.module, self.params

        @functools.partial(jax.jit)
        def fwd(hist):
            return mod.apply(prm, (hist - mean) / std, method="act")

        self._fwd = fwd

    def __call__(self, hist_raw):
        return np.asarray(self._fwd(jnp.asarray(hist_raw, jnp.float32)))
