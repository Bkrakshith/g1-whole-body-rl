#!/usr/bin/env python
"""STUB reference motions for dry-running Stage C before the ULTRA data is
downloaded: the robot stands at home while its upper body moves through
feasible teleop poses (configs/upper_body_poses.yaml) with min-jerk timing.
Robot-scale data -> use ref_root_scale 1.0 with it (train_tracking --dry_run
does). Written to data/motions_synthetic/.
"""
import os
import sys

import mujoco
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from g1_wbc.config import MJX_XML, RL_ROOT, load_config  # noqa: E402
from g1_wbc.layout import UPPER_IDX  # noqa: E402
from g1_wbc.motion_data import prepare, save_npz  # noqa: E402


def main(n=8, seconds=12.0, fps=50.0):
    cfg = load_config("stage_c_tracking.yaml")
    model = mujoco.MjModel.from_xml_path(MJX_XML)
    home = model.key("home").qpos
    poses = np.array([p["q"] for p in yaml.safe_load(open(os.path.join(
        RL_ROOT, "configs", "upper_body_poses.yaml")))["library"]])
    rng = np.random.default_rng(0)
    out = os.path.join(RL_ROOT, "data", "motions_synthetic")
    os.makedirs(out, exist_ok=True)
    T = int(seconds * fps)
    for i in range(n):
        dof = np.tile(home[7:], (T, 1))
        cur, t = dof[0, list(UPPER_IDX)].copy(), 0
        while t < T:
            p = poses[rng.integers(len(poses))]
            mv = int(max(0.5, np.abs(p - cur).max() / 1.5) * fps)
            for k in range(min(mv, T - t)):
                u = (k + 1) / mv
                dof[t + k, list(UPPER_IDX)] = cur + (u ** 3 * (10 - 15 * u + 6 * u * u)) * (p - cur)
            t += mv
            hold = int(rng.uniform(0.5, 1.5) * fps)
            dof[t:t + hold, list(UPPER_IDX)] = p
            t += hold
            cur = p
        m = {"fps": fps, "root_pos": np.tile(home[0:3], (T, 1)),
             "root_quat": np.tile(home[3:7], (T, 1)), "dof": dof}
        r = prepare(m, model, cfg.train.tracking.key_bodies, fps,
                    int(cfg.train.tracking.standing_prefix_frames))
        save_npz(os.path.join(out, f"synthetic_{i:03d}.npz"), r)
    print(f"{n} STUB motions -> {out}")


if __name__ == "__main__":
    main()
