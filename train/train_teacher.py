#!/usr/bin/env python
"""Train the privileged teacher through a curriculum (Stage A 'stand' or
Stage B 'teleop').

  ./run.sh train/train_teacher.py --curriculum stand  --run_name stand_v1 --backend gpu
  ./run.sh train/train_teacher.py --curriculum teleop --run_name teleop_v1 --backend gpu \
        --init_from outputs/checkpoints/stand_v1/ckpt_N --start_stage b1_walk_calm
  ./run.sh train/train_teacher.py --curriculum teleop --run_name smoke --dry_run

--dry_run: the REAL brax training scan (catches pytree / metric-key / shape
mismatches unit tests cannot see, skill Phase 3) on the FINAL stage of the
curriculum with tiny sizes: 64 envs, a few PPO iterations, gates off. It is
NOT training. Long runs: detached + pinned (run.sh pins CPUs), see README.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from g1_wbc.backend import select_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--curriculum", choices=["stand", "teleop"], default="teleop")
    ap.add_argument("--num_envs", type=int, default=None)
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--start_stage", default=None)
    ap.add_argument("--stop_after", default=None, help="stop after this stage passes")
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dry_run_iters", type=int, default=3)
    ap.add_argument("--ppo_bench", action="store_true",
                    help="THROUGHPUT BENCHMARK, not training: the real PPO geometry (num_envs from "
                         "train.yaml) for --dry_run_iters iterations, prints env steps/s")
    a = ap.parse_args()
    select_backend(a.backend)
    from g1_wbc.config import _deep_merge, load_config
    from g1_wbc.ppo_teacher import train_teacher
    cfg = load_config(a.overlay)
    if a.ppo_bench:
        p = cfg.train.ppo
        final = dict(cfg.train.curricula[a.curriculum][-1])
        per_it = int(p.batch_size) * int(p.num_minibatches) * int(p.unroll_length)
        steps = (a.dry_run_iters + 1) * per_it
        final.update(num_timesteps=steps, gate_tracking=0.0, gate_air_time=0.0,
                     gate_survival=0.0, gate_upper=0.0)
        cfg.train = _deep_merge(cfg.train, {"curricula": {a.curriculum: [final]},
                                            "checkpoint": {"eval_every_steps": steps // 2,
                                                           "root": "outputs/dry_run"}})
        print(f"[teacher] PPO THROUGHPUT BENCHMARK (not training): {p.num_envs} envs, "
              f"{steps:,} steps", flush=True)
    elif a.dry_run:
        final = dict(cfg.train.curricula[a.curriculum][-1])
        n_env, unroll = 64, 20
        steps = a.dry_run_iters * n_env * unroll * 4   # batch 32 x 4 minibatches = 2 x 64 envs
        final.update(num_timesteps=steps, gate_tracking=0.0, gate_air_time=0.0,
                     gate_survival=0.0, gate_upper=0.0)
        cfg.train = _deep_merge(cfg.train, {
            "curricula": {a.curriculum: [final]},
            "ppo": {"num_envs": n_env, "num_eval_envs": 16, "batch_size": 32,
                    "num_minibatches": 4, "episode_length": 100},
            "checkpoint": {"eval_every_steps": steps // 2, "root": "outputs/dry_run"}})
        print(f"[teacher] DRY RUN on '{final['name']}': {n_env} envs, {steps:,} steps, gates off",
              flush=True)
    if bool(a.init_from) != bool(a.start_stage):
        ap.error("--init_from and --start_stage go together")
    train_teacher(a.run_name, a.curriculum, a.num_envs, a.resume, cfg, a.init_from,
                  a.start_stage, a.stop_after)


if __name__ == "__main__":
    main()
