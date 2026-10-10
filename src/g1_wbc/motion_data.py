"""Stage C reference motions: ULTRA .pt loader -> our 50 Hz .npz -> padded
JAX motion library.

ULTRA [T, 630] layout (read in ultra/env/tasks/ultra_g1.py _load_motion, 30 fps,
interpolated x2 to 60 fps there):
   0:3    root position (m, world)
   3:7    root rotation quaternion (x, y, z, w; IsaacGym convention)
   9:162  dof positions, padded to 153; the first 29 are the G1 joints in the
          contract order (URDF DFS = legs, waist, arms; = ULTRA init_dof order)
 162:318  52 x 3 body positions, 318:325 object pos/rot, 330 object contact,
 331:383  body contacts, 383:591 body rotations (unused here)
Only root pose + dof are used; key-body positions/velocities are recomputed
by forward kinematics on OUR model, so they are exact for the g1-brainco-description build
regardless of the archive's body ordering.
"""

import glob
import os

import mujoco
import numpy as np

from .layout import JOINT_NAMES

ULTRA_FPS = 30.0


def load_ultra_pt(path):
    import torch
    x = torch.load(path, map_location="cpu")
    x = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, np.float64)[1:]
    if x.ndim != 2 or x.shape[1] < 38:
        raise ValueError(f"{path}: expected [T, 630] ULTRA motion, got {x.shape}")
    xyzw = x[:, 3:7]
    return {"fps": ULTRA_FPS, "root_pos": x[:, 0:3], "root_quat": xyzw[:, [3, 0, 1, 2]],
            "dof": x[:, 9:9 + 29]}


def _slerp_nlerp(q0, q1, u):
    q1 = np.where((np.sum(q0 * q1, -1, keepdims=True) < 0), -q1, q1)
    q = (1 - u) * q0 + u * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample(m, fps_out):
    T = m["root_pos"].shape[0]
    t_in = np.arange(T) / m["fps"]
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fps_out)
    i = np.clip(np.searchsorted(t_in, t_out, side="right") - 1, 0, T - 2)
    u = ((t_out - t_in[i]) * m["fps"])[:, None]
    lerp = lambda a: a[i] * (1 - u) + a[i + 1] * u  # noqa: E731
    return {"fps": fps_out, "root_pos": lerp(m["root_pos"]), "dof": lerp(m["dof"]),
            "root_quat": _slerp_nlerp(m["root_quat"][i], m["root_quat"][i + 1], u)}


def forward_kinematics(model, m, key_bodies):
    d = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES]
    kid = [model.body(b).id for b in key_bodies]
    T = m["root_pos"].shape[0]
    out = np.zeros((T, len(kid), 3))
    for t in range(T):
        d.qpos[0:3] = m["root_pos"][t]
        d.qpos[3:7] = m["root_quat"][t]
        d.qpos[qadr] = m["dof"][t]
        mujoco.mj_kinematics(model, d)
        out[t] = d.xpos[kid]
    return out


def prepare(m, model, key_bodies, fps_out, prefix_frames):
    """ULTRA-style: resample, prepend `prefix_frames` copies of frame 0
    (ULTRA prepends 30 standing frames), FK key bodies."""
    r = resample(m, fps_out) if abs(m["fps"] - fps_out) > 1e-6 else dict(m)
    for k in ("root_pos", "root_quat", "dof"):
        r[k] = np.concatenate([np.repeat(r[k][:1], prefix_frames, 0), r[k]], 0)
    r["key_pos"] = forward_kinematics(model, r, key_bodies)
    r["fps"] = fps_out
    return r


def save_npz(path, r):
    np.savez_compressed(path, fps=r["fps"], root_pos=r["root_pos"].astype(np.float32),
                        root_quat=r["root_quat"].astype(np.float32),
                        dof=r["dof"].astype(np.float32), key_pos=r["key_pos"].astype(np.float32))


class MotionLib:
    """Padded (N, Tmax, ...) arrays of every prepared motion (numpy); the env
    moves them to the device. Velocities by finite differences."""

    def __init__(self, motion_dir, max_motions=None):
        files = sorted(glob.glob(os.path.join(motion_dir, "*.npz")))[: max_motions or None]
        if not files:
            raise FileNotFoundError(
                f"no motions in {motion_dir}: run scripts/prepare_ultra_motions.py (real data) or "
                "scripts/make_synthetic_motion.py (stub)")
        ms = [dict(np.load(f)) for f in files]
        self.files = files
        self.fps = float(ms[0]["fps"])
        self.lengths = np.array([m["dof"].shape[0] for m in ms], np.int32)
        T = int(self.lengths.max())

        def pad(key):
            a = np.stack([np.concatenate([m[key], np.repeat(m[key][-1:], T - len(m[key]), 0)])
                          for m in ms]).astype(np.float32)
            return a

        self.root_pos, self.root_quat = pad("root_pos"), pad("root_quat")
        self.dof, self.key_pos = pad("dof"), pad("key_pos")
        dt = 1.0 / self.fps
        self.key_vel = np.concatenate([np.zeros_like(self.key_pos[:, :1]),
                                       np.diff(self.key_pos, axis=1) / dt], 1)
        self.root_vel = np.concatenate([np.zeros_like(self.root_pos[:, :1]),
                                        np.diff(self.root_pos, axis=1) / dt], 1)
        self.dof_vel = np.concatenate([np.zeros_like(self.dof[:, :1]),
                                       np.diff(self.dof, axis=1) / dt], 1)
