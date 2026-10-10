#!/usr/bin/env python
"""Distil a trained teacher into the deployable CVAE student (DAgger).

  ./run.sh train/train_student.py --teacher_ckpt outputs/checkpoints/teleop_v1/ckpt_N \
        --curriculum teleop --run_name teleop_student_v1 --backend gpu
  ./run.sh train/train_student.py --teacher_ckpt <ckpt> --run_name smoke --dry_run

--dry_run: a few distillation iterations with 64 envs (proves the loop; NOT training).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from g1_wbc.backend import select_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--curriculum", choices=["stand", "teleop", "tracking"], default="teleop")
    ap.add_argument("--num_envs", type=int, default=None)
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    select_backend(a.backend)
    from g1_wbc.config import RL_ROOT, _deep_merge, load_config
    from g1_wbc.distill_student import distill
    cfg = load_config(a.overlay)
    kw = {}
    if a.dry_run:
        cfg.train = _deep_merge(cfg.train, {"distill": {"beta_hold_iters": 1, "beta_decay_iters": 2,
                                                        "schedule_iters": [0, 4]},
                                            "ppo": {"episode_length": 100}})
        kw = dict(num_envs=64, num_iterations=6, out_root=os.path.join(RL_ROOT, "outputs", "dry_run"))
        print("[student] DRY RUN: 64 envs, 6 iterations", flush=True)
    env_fn = None
    if a.curriculum == "tracking":      # Stage C: needs --overlay stage_c_tracking.yaml
        from g1_wbc.tracking_env import G1TrackingEnv
        stage = cfg.stages("tracking")[-1]
        env_fn = lambda c: G1TrackingEnv(c, stage, student_noise=True, emit_clean_state=True)  # noqa: E731
    distill(a.teacher_ckpt, a.run_name, cfg, curriculum_name=a.curriculum, env_fn=env_fn,
            **{"num_envs": a.num_envs, **kw})


if __name__ == "__main__":
    main()
