#!/usr/bin/env python
"""Upper-body frames (waist 3 + arms 14) from prepared ULTRA motions ->
data/upper_body_bank.npz, an extra source of realistic teleop targets for
Stage B (commands.yaml upper_body.p_motion_bank > 0 enables it). Frames
outside the feasible envelope (configs/upper_body_poses.yaml box, joint
limits) are dropped.

  ./run.sh scripts/build_motion_bank.py [--motions data/motions] [--stride 5]
"""
import argparse
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from g1_wbc.config import RL_ROOT  # noqa: E402
from g1_wbc.layout import UPPER_IDX  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", default="data/motions")
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    env = yaml.safe_load(open(os.path.join(RL_ROOT, "configs", "upper_body_poses.yaml")))
    lo, hi = np.array(env["box_lo"]), np.array(env["box_hi"])
    frames = []
    for f in sorted(glob.glob(os.path.join(RL_ROOT, a.motions, "*.npz"))):
        frames.append(np.load(f)["dof"][:: a.stride][:, list(UPPER_IDX)])
    if not frames:
        raise SystemExit("no prepared motions; run scripts/prepare_ultra_motions.py first")
    x = np.concatenate(frames)
    keep = np.all((x >= lo - 0.2) & (x <= hi + 0.2), axis=1)
    out = os.path.join(RL_ROOT, "data", "upper_body_bank.npz")
    np.savez_compressed(out, upper_q=x[keep].astype(np.float32))
    print(f"{keep.sum()}/{len(x)} frames inside the envelope -> {out}")


if __name__ == "__main__":
    main()
