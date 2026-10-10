#!/usr/bin/env python
"""Convert ULTRA's retargeted+augmented G1 motions ([T, 630] .pt, 30 fps) to
our 50 Hz .npz (root pose, 29 dof, FK key bodies on OUR model, ULTRA's
standing prefix). Needs the archive extracted first (configs/stage_c_tracking.yaml
has the link; nothing is downloaded here).

  ./run.sh scripts/prepare_ultra_motions.py --src data/ultra/OMOMO_retarget_aug --out data/motions
"""
import argparse
import glob
import os
import sys

import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from g1_wbc.config import MJX_XML, RL_ROOT, load_config  # noqa: E402
from g1_wbc.motion_data import load_ultra_pt, prepare, save_npz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/ultra/OMOMO_retarget_aug")
    ap.add_argument("--out", default="data/motions")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config("stage_c_tracking.yaml")
    tc = cfg.train.tracking
    fps = 1.0 / (cfg.actuators.control.physics_dt * cfg.actuators.control.decimation)
    model = mujoco.MjModel.from_xml_path(MJX_XML)
    src = os.path.join(RL_ROOT, a.src)
    files = sorted(glob.glob(os.path.join(src, "**", "*.pt"), recursive=True))[: a.limit]
    if not files:
        raise SystemExit(f"no .pt files under {src}; download + extract the ULTRA archive first")
    out = os.path.join(RL_ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    for f in files:
        r = prepare(load_ultra_pt(f), model, tc.key_bodies, fps, int(tc.standing_prefix_frames))
        save_npz(os.path.join(out, os.path.splitext(os.path.basename(f))[0] + ".npz"), r)
    print(f"{len(files)} motions -> {out}")


if __name__ == "__main__":
    main()
