#!/usr/bin/env python
"""Stage C: ULTRA motion-tracking teacher (configs/stage_c_tracking.yaml).

  ./run.sh train/train_tracking.py --run_name track_v1 --backend gpu      # needs data/motions
  ./run.sh train/train_tracking.py --run_name smoke --dry_run             # STUB motions, tiny
Distil afterwards with train/train_student.py --curriculum tracking --overlay stage_c_tracking.yaml
(the student reads the same 'state' layout: proprio + reference goals).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from g1_wbc.backend import select_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--num_envs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    select_backend(a.backend)
    from g1_wbc.config import RL_ROOT, _deep_merge, load_config
    from g1_wbc.ppo_teacher import train_teacher
    from g1_wbc.tracking_env import G1TrackingEnv
    cfg = load_config("stage_c_tracking.yaml")
    if a.dry_run:
        if not os.path.isdir(os.path.join(RL_ROOT, "data", "motions_synthetic")):
            raise SystemExit("run scripts/make_synthetic_motion.py first")
        final = dict(cfg.train.curricula.tracking[-1])
        steps = 3 * 64 * 32 * 2
        final.update(num_timesteps=steps, gate_survival=0.0)
        cfg.train = _deep_merge(cfg.train, {
            "curricula": {"tracking": [final]},
            "tracking": {"motion_dir": "data/motions_synthetic", "ref_root_scale": 1.0},
            "ppo": {"num_envs": 64, "num_eval_envs": 16, "batch_size": 32, "num_minibatches": 4,
                    "episode_length": 100},
            "checkpoint": {"eval_every_steps": steps // 2, "root": "outputs/dry_run"}})
        print(f"[tracking] DRY RUN on STUB motions: 64 envs, {steps:,} steps", flush=True)
    train_teacher(a.run_name, "tracking", a.num_envs, a.resume, cfg,
                  env_fn=lambda c, s: G1TrackingEnv(c, s))


if __name__ == "__main__":
    main()
