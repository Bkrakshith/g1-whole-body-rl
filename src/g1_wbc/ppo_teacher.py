"""Privileged teacher via brax PPO through a staged curriculum. Ported from
the validated AgiBot X1 driver (Mujoco_agibot/src/x1_locomotion/ppo_teacher.py):
one brax ppo.train per stage, params carried across stages, checkpoint at
every eval, the next stage warm-starts from the BEST passing checkpoint (L5),
never-varied normaliser dims reset at every warm start (M12), gate metrics
printed and appended to evals.jsonl (M17)."""

import dataclasses
import json
import os
import time

import jax
import numpy as np
from brax.training.agents.ppo import train as ppo

from . import curriculum
from .config import RL_ROOT, Config, Stage, load_config
from .env import G1TeleopEnv
from .networks import make_networks_factory
from .wrappers import wrap_for_training


def ckpt_dir(cfg: Config, run_name: str) -> str:
    return os.path.join(RL_ROOT, cfg.train.checkpoint.root, run_name)


def save_checkpoint(path, params, meta):
    import orbax.checkpoint as ocp
    path = os.path.abspath(path)
    ocp.PyTreeCheckpointer().save(path, jax.tree.map(np.asarray, params), force=True)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(path):
    import orbax.checkpoint as ocp
    ck = ocp.PyTreeCheckpointer()
    path = os.path.abspath(path.rstrip("/"))
    meta_tree = ck.metadata(path)
    meta_tree = getattr(meta_tree, "item_metadata", meta_tree)
    args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta_tree)
    params = ck.restore(path, restore_args=args)
    with open(path + ".meta.json") as f:
        return params, json.load(f)


def latest_checkpoint(run_dir):
    if not os.path.isdir(run_dir):
        return None
    c = [d for d in os.listdir(run_dir) if d.startswith("ckpt_") and d.split("_")[-1].isdigit()
         and os.path.exists(os.path.join(run_dir, d + ".meta.json"))]
    return os.path.join(run_dir, max(c, key=lambda d: int(d.split("_")[-1]))) if c else None


def rehydrate_normalizer(d):
    from brax.training.acme.running_statistics import RunningStatisticsState
    from brax.training.types import UInt64
    if isinstance(d, RunningStatisticsState):
        return d
    count = d["count"]
    if isinstance(count, dict):
        count = UInt64(hi=count["hi"], lo=count["lo"])
    return RunningStatisticsState(count=count, mean=d["mean"], summed_variance=d["summed_variance"],
                                  std=d["std"], std_eps=d.get("std_eps", 0.0))


_FROZEN_STD = 1e-3


def unfreeze_constant_dims(norm):
    """Skill M12: a privileged dim that never varied in the previous stage
    (e.g. every DR latent in a1_stand at dr_scale 0) sits at brax's std floor
    1e-6; the first real value would normalise to ~1e5. Reset such dims to
    unit std before warm-starting (summed_variance = count)."""
    import jax.numpy as jnp
    from brax.training.acme.running_statistics import RunningStatisticsState
    c = norm.count
    n = (float(c.hi) * 2.0 ** 32 + float(c.lo)) if hasattr(c, "hi") else float(c)
    if n <= 0:
        return norm
    std, sv = {}, {}
    for k in norm.std:
        frozen = norm.std[k] < _FROZEN_STD
        std[k] = jnp.where(frozen, 1.0, norm.std[k])
        sv[k] = jnp.where(frozen, n, norm.summed_variance[k])
    return RunningStatisticsState(count=c, mean=norm.mean, summed_variance=sv, std=std,
                                  std_eps=getattr(norm, "std_eps", 0.0))


def as_restore_tuple(params):
    params = tuple(params)
    return (unfreeze_constant_dims(rehydrate_normalizer(params[0])),) + params[1:]


def train_stage(cfg: Config, stage: Stage, stage_idx: int, run_name: str, stages: list,
                num_envs=None, restore_params=None, seed=0, env_fn=None):
    p = cfg.train.ppo
    curriculum.EPISODE_STEPS = int(p.episode_length)
    num_envs = num_envs or int(p.num_envs)
    env = (env_fn or (lambda c, s: G1TeleopEnv(c, s)))(cfg, stage)
    run_dir = ckpt_dir(cfg, run_name)
    os.makedirs(run_dir, exist_ok=True)
    num_evals = max(2, int(stage.num_timesteps // cfg.train.checkpoint.eval_every_steps))
    steps_offset = sum(s.num_timesteps for s in stages[:stage_idx])
    t0 = time.time()
    last = {}

    def progress_fn(num_steps, metrics):
        nonlocal last
        if metrics:
            last = dict(metrics)
        ok, g = curriculum.gate_passed(stage, metrics) if metrics else (False, {})
        sps = num_steps / max(time.time() - t0, 1e-6)
        print(f"[{stage.name}] steps={num_steps:,} reward={metrics.get('eval/episode_reward', float('nan')):.1f} "
              f"ep_len={metrics.get('eval/avg_episode_length', float('nan')):.0f} "
              + " ".join(f"{k}={v:.3f}" for k, v in g.items())
              + f" pass={ok} ({sps:,.0f} steps/s)", flush=True)
        if metrics:
            with open(os.path.join(run_dir, "evals.jsonl"), "a") as fh:
                fh.write(json.dumps({"stage": stage.name, "global_step": steps_offset + int(num_steps),
                                     **{k: float(v) for k, v in metrics.items()}}) + "\n")

    def policy_params_fn(current_step, make_policy, params):
        del make_policy
        save_checkpoint(os.path.join(run_dir, f"ckpt_{steps_offset + current_step}"), params,
                        meta={"stage": stage_idx, "stage_name": stage.name,
                              "stage_step": int(current_step), "seed": seed, "run_name": run_name,
                              "stage_cfg": dataclasses.asdict(stage), "wall_time": time.time() - t0})

    _, params, metrics = ppo.train(
        environment=env, wrap_env_fn=wrap_for_training,
        num_timesteps=stage.num_timesteps, episode_length=int(p.episode_length),
        num_envs=num_envs, num_eval_envs=int(p.num_eval_envs), num_evals=num_evals,
        unroll_length=int(p.unroll_length), batch_size=int(p.batch_size),
        num_minibatches=int(p.num_minibatches), num_updates_per_batch=int(p.num_updates_per_batch),
        learning_rate=float(p.learning_rate), entropy_cost=float(p.entropy_cost),
        discounting=float(p.discounting), gae_lambda=float(p.gae_lambda),
        reward_scaling=float(p.reward_scaling), clipping_epsilon=float(p.clipping_epsilon),
        max_grad_norm=float(p.max_grad_norm), normalize_observations=bool(p.normalize_observations),
        network_factory=make_networks_factory(cfg), progress_fn=progress_fn,
        policy_params_fn=policy_params_fn, restore_params=restore_params, seed=seed + stage_idx)
    last.update(metrics or {})
    return params, last


def best_passing_checkpoint(run_dir, stage: Stage):
    """Among this stage's evals that clear every gate, the checkpoint with the
    longest mean episode (L5: select, don't take the last)."""
    path = os.path.join(run_dir, "evals.jsonl")
    if not os.path.exists(path):
        return None, None
    best = None
    for line in open(path):
        row = json.loads(line)
        if row.get("stage") != stage.name:
            continue
        ok, _ = curriculum.gate_passed(stage, row)
        ck = os.path.join(run_dir, f"ckpt_{row['global_step']}")
        if ok and os.path.isdir(ck):
            key = row.get("eval/avg_episode_length", 0.0)
            if best is None or key > best[0]:
                best = (key, ck, row)
    return (best[1], best[2]) if best else (None, None)


def train_teacher(run_name, curriculum_name="teleop", num_envs=None, resume=False, cfg=None,
                  init_from=None, start_stage=None, stop_after=None, env_fn=None):
    cfg = cfg or load_config()
    stages = cfg.stages(curriculum_name)
    names = [s.name for s in stages]
    seed = int(cfg.train.ppo.seed)
    run_dir = ckpt_dir(cfg, run_name)
    start, restore = 0, None
    if init_from:
        params, meta = load_checkpoint(init_from)
        restore = as_restore_tuple(params)
        if start_stage not in names:
            raise SystemExit(f"--start_stage must be one of {names}")
        start = names.index(start_stage)
        print(f"--init_from {init_from} (stage '{meta.get('stage_name')}') -> '{start_stage}'")
    elif resume:
        latest = latest_checkpoint(run_dir)
        if latest:
            params, meta = load_checkpoint(latest)
            if meta.get("stage_name") not in names:
                raise SystemExit(f"checkpoint stage {meta.get('stage_name')} not in {names}")
            start, seed = names.index(meta["stage_name"]), int(meta["seed"])
            restore = as_restore_tuple(params)
            print(f"--resume: {latest} (stage '{meta['stage_name']}')")
    for i in range(start, len(stages)):
        st = stages[i]
        print(f"\n=== stage {i + 1}/{len(stages)}: {st.name} ({st.num_timesteps:,} steps) ===", flush=True)
        params, metrics = train_stage(cfg, st, i, run_name, stages, num_envs, restore, seed, env_fn)
        restore = as_restore_tuple(params)
        best, row = best_passing_checkpoint(run_dir, st)
        if best is None:
            _, g = curriculum.gate_passed(st, metrics)
            print(curriculum.stall_report(st, g))
            raise SystemExit(2)
        restore = as_restore_tuple(load_checkpoint(best)[0])
        print(f"[{st.name}] gates passed at {os.path.basename(best)} -> next stage warm-starts from it")
        if stop_after and st.name == stop_after:
            break
    print(f"\nTeacher done. Checkpoints: {run_dir}")
    return restore
