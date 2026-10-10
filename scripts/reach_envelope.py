#!/usr/bin/env python
"""Static feasibility analysis of the TELEOP upper-body target envelope
(skill reach-manifold-analysis.md, adapted: the arms move, the feet stay
planted, the legs hold the home pose). Writes configs/upper_body_poses.yaml,
which the Stage B env samples its upper-body targets from.

For every candidate pose (waist 3 + left arm 7 + right arm 7), on the
full-fidelity eval model (mesh collisions):
  1. self-collision: no NEW penetrating contact vs the home pose (arm through
     torso / thigh / other arm);
  2. static balance: normalised CoM margin m_hat = sd(CoM_xy, support
     polygon) / m_max > m_hat_min (skill theory 9; polygon = hull of the 8
     sole spheres, m_max by grid search);
  3. static gravity torques (qfrc_bias at qvel = 0) within the PD torque clip;
and a per-joint feasible box is grown/shrunk from a hand-chosen teleop box by
uniform sampling until >= feasible_target of samples pass 1-3.

The pelvis stays level in this analysis (legs at home), so the tilt the task
needs is ~0 and the tilt termination (0.8 rad) has >= 40 deg headroom; the
CoM margin is the binding quantity (reported per pose).

  ./run.sh scripts/reach_envelope.py [--samples 3000]
"""

import argparse
import os
import sys

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(RL, "src"))

from g1_wbc.layout import JOINT_NAMES, UPPER_IDX, torque_limit  # noqa: E402

# Left arm poses (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
# wrist_roll, wrist_pitch, wrist_yaw). G1 signs: shoulder_pitch < 0 = arm
# forward/up, left shoulder_roll > 0 = abduct (out), elbow > 0 = flex.
# The right arm is the mirror (roll/yaw axes negate).
ARM_POSES = {
    "default":        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
    "reach_low":      [-0.6, 0.15, 0.0, 0.6, 0.0, 0.0, 0.0],
    "reach_mid":      [-1.2, 0.15, 0.0, 0.3, 0.0, 0.0, 0.0],
    "reach_high":     [-1.8, 0.15, 0.0, 0.2, 0.0, 0.0, 0.0],
    "overhead":       [-2.6, 0.25, 0.0, 0.3, 0.0, 0.0, 0.0],
    "side_raise":     [0.0, 1.3, 0.0, 0.3, 0.0, 0.0, 0.0],
    "wave":           [-0.4, 1.0, 0.0, 1.6, 0.0, 0.0, 0.0],
    "hand_to_chest":  [-0.6, 0.35, -0.4, 1.7, 0.0, 0.0, 0.0],
    "table_pick":     [-0.9, 0.2, 0.2, 1.0, 0.0, -0.6, 0.0],
    "hold_box":       [-0.8, 0.05, 0.4, 1.2, 0.0, 0.0, 0.0],
    "carry_tray":     [-0.4, 0.2, 0.0, 1.5, 1.2, 0.0, 0.0],
    "palm_up_reach":  [-1.0, 0.2, 0.0, 0.5, 1.5, 0.0, 0.0],
    "point_side":     [-0.8, 0.9, 0.0, 0.1, 0.0, 0.0, 0.0],
    "arm_back":       [0.6, 0.2, 0.0, 0.4, 0.0, 0.0, 0.0],
    "wrist_flex":     [-0.7, 0.2, 0.0, 1.0, 0.0, 1.2, 0.8],
}
MIRROR_ARM = np.array([1, -1, -1, 1, -1, 1, -1], float)
WAIST_POSES = {"neutral": [0.0, 0.0, 0.0], "turn_l": [0.5, 0.0, 0.0],
               "turn_r": [-0.5, 0.0, 0.0], "lean_fwd": [0.0, 0.0, 0.3],
               "lean_side": [0.0, 0.15, 0.0]}
MIRROR_WAIST = np.array([-1, -1, 1], float)

# Starting teleop box (waist 3, left arm 7); right arm = mirror.
BOX0 = {
    "waist_yaw_joint": (-0.8, 0.8), "waist_roll_joint": (-0.25, 0.25),
    "waist_pitch_joint": (-0.2, 0.45),
    "left_shoulder_pitch_joint": (-2.8, 1.0), "left_shoulder_roll_joint": (-0.3, 2.0),
    "left_shoulder_yaw_joint": (-1.3, 1.3), "left_elbow_joint": (-0.9, 2.0),
    "left_wrist_roll_joint": (-1.9, 1.9), "left_wrist_pitch_joint": (-1.5, 1.5),
    "left_wrist_yaw_joint": (-1.5, 1.5),
}


def mirror_left_to_right(lo, hi):
    """Joint box of the right arm from the left one (roll/yaw negate)."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    rlo = np.where(MIRROR_ARM > 0, lo, -hi)
    rhi = np.where(MIRROR_ARM > 0, hi, -lo)
    return rlo, rhi


class Checker:
    def __init__(self, xml, act, m_hat_min):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.home = self.m.key("home").qpos.copy()
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINT_NAMES])
        self.dadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in JOINT_NAMES])
        self.lo = self.m.jnt_range[[self.m.joint(n).id for n in JOINT_NAMES], 0]
        self.hi = self.m.jnt_range[[self.m.joint(n).id for n in JOINT_NAMES], 1]
        self.tau_lim = np.asarray(torque_limit(act))
        self.m_hat_min = m_hat_min
        self.floor = self.m.geom("floor").id
        self.pairs = self._collision_pairs()
        self._set(self.home)
        mujoco.mj_forward(self.m, self.d)
        self.base_pairs = self._pairs()
        soles = [g for g in range(self.m.ngeom)
                 if (self.m.geom(g).name or "").startswith(("left_sole_", "right_sole_"))]
        pts = self.d.geom_xpos[soles][:, :2]
        self.hull = _hull(pts)
        self.m_max = _m_max(self.hull)

    def _set(self, q):
        self.d.qpos[:] = q
        self.d.qvel[:] = 0.0
        self.d.qacc[:] = 0.0

    def _collision_pairs(self):
        """(arm geom, other geom) pairs to test. The g1-brainco-description MJCF disables
        self-collision (contype 2 / conaffinity 1 on every body geom), so the
        simulator's contact list is empty by construction; we measure
        geometry directly with mj_geomDistance on the collision meshes."""
        m = self.m
        coll = [g for g in range(m.ngeom) if m.geom_group[g] == 3
                and m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX,
                                       mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER,
                                       mujoco.mjtGeom.mjGEOM_SPHERE)
                and not (m.geom(g).name or "").startswith(("left_sole_", "right_sole_"))]
        body = lambda g: m.body(m.geom_bodyid[g]).name
        def side_arm(g, side):
            n = body(g)
            return n.startswith(side) and any(k in n for k in ("shoulder_yaw", "elbow", "wrist", "hand", "base", "thumb", "index", "middle", "ring", "pinky"))
        pairs = []
        for side, other in (("left", "right"), ("right", "left")):
            arm = [g for g in coll if side_arm(g, side)]
            rest = [g for g in coll if not side_arm(g, side)
                    and not body(g).startswith(f"{side}_shoulder")]
            pairs += [(a, b) for a in arm for b in rest if a < b or not side_arm(b, other)]
        return sorted(set((min(a, b), max(a, b)) for a, b in pairs))

    def _pairs(self, margin=0.0):
        out = set()
        ft = np.zeros(6)
        for a, b in self.pairs:
            if mujoco.mj_geomDistance(self.m, self.d, a, b, 0.05, ft) < -margin:
                out.add((a, b))
        return out

    def check(self, upper):
        q = self.home.copy()
        q[self.qadr[list(UPPER_IDX)]] = np.clip(upper, self.lo[list(UPPER_IDX)],
                                                self.hi[list(UPPER_IDX)])
        self._set(q)
        mujoco.mj_forward(self.m, self.d)
        new = self._pairs() - self.base_pairs
        com = self.d.subtree_com[1][:2]
        m_hat = _signed_dist(com, self.hull) / self.m_max
        # qvel = 0 -> qfrc_bias is exactly the gravity torque each joint must
        # hold statically (mj_inverse after mj_forward would read back the
        # forward qacc and return ~0).
        tau = self.d.qfrc_bias[self.dadr]
        # the legs carry the whole body statically; only the upper body's own
        # holding torques are a property of the pose
        tau_ratio = np.max(np.abs(tau[list(UPPER_IDX)]) / self.tau_lim[list(UPPER_IDX)])
        ok = (not new) and m_hat > self.m_hat_min and tau_ratio < 1.0
        return ok, {"collision": bool(new), "m_hat": float(m_hat),
                    "tau_ratio": float(tau_ratio),
                    "com_xy": com.tolist()}


def _hull(pts):
    pts = sorted(map(tuple, pts))

    def half(points):
        out = []
        for p in points:
            while len(out) >= 2 and _cross2(np.subtract(out[-1], out[-2]),
                                            np.subtract(p, out[-2])) <= 0:
                out.pop()
            out.append(p)
        return out
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _signed_dist(p, hull):
    """+ inside, - outside (distance to the nearest edge)."""
    n = len(hull)
    d = np.inf
    inside = True
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = b - a
        t = np.clip(np.dot(p - a, e) / np.dot(e, e), 0, 1)
        d = min(d, np.linalg.norm(p - (a + t * e)))
        if _cross2(e, p - a) < 0:
            inside = False
    return d if inside else -d


def _m_max(hull):
    lo, hi = hull.min(0), hull.max(0)
    xs, ys = np.linspace(lo[0], hi[0], 61), np.linspace(lo[1], hi[1], 61)
    return max(_signed_dist(np.array([x, y]), hull) for x in xs for y in ys)


def upper_vec(waist, left, right):
    return np.concatenate([waist, left, right])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--m_hat_min", type=float, default=0.2)
    ap.add_argument("--feasible_target", type=float, default=0.85)
    ap.add_argument("--xml", default=os.path.join(RL, "assets", "g1_eval.xml"))
    ap.add_argument("--out", default=os.path.join(RL, "configs", "upper_body_poses.yaml"))
    args = ap.parse_args()
    with open(os.path.join(RL, "configs", "actuators.yaml")) as f:
        act = yaml.safe_load(f)
    ck = Checker(args.xml, act, args.m_hat_min)
    print(f"support polygon m_max {ck.m_max * 100:.1f} cm; home CoM margin "
          f"{_signed_dist(ck.d.subtree_com[1][:2], ck.hull) * 100:.1f} cm")

    # ---------------------------------------------------- 1. library poses
    library, report = [], []
    rng = np.random.default_rng(0)
    names = list(ARM_POSES)
    combos = [(w, l, r) for w in WAIST_POSES for l in names for r in names
              if (l == r) or (l == "default") or (r == "default") or rng.uniform() < 0.08]
    for w, ln, rn in combos:
        left = np.array(ARM_POSES[ln])
        right = np.array(ARM_POSES[rn]) * MIRROR_ARM
        waist = np.array(WAIST_POSES[w])
        u = upper_vec(waist, left, right)
        ok, info = ck.check(u)
        report.append({"waist": w, "left": ln, "right": rn, "ok": ok, **info})
        if ok:
            library.append({"name": f"{w}|{ln}|{rn}", "q": [round(float(v), 4) for v in u],
                            "m_hat": round(info["m_hat"], 3)})
    n_ok = sum(r["ok"] for r in report)
    print(f"library: {n_ok}/{len(report)} candidate poses feasible")
    for r in report:
        if not r["ok"] and r["waist"] == "neutral":
            print(f"  rejected {r['left']:>14s} | {r['right']:<14s} collision={r['collision']} "
                  f"m_hat={r['m_hat']:+.2f} tau={r['tau_ratio']:.2f}")

    # ---------------------------------------------------- 2. feasible box
    wnames = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    lnames = [n for n in BOX0 if n.startswith("left_")]
    lo = np.array([BOX0[n][0] for n in wnames + lnames])
    hi = np.array([BOX0[n][1] for n in wnames + lnames])
    for it in range(10):
        s = rng.uniform(lo, hi, (args.samples, len(lo)))
        ok = np.zeros(args.samples, bool)
        for i, x in enumerate(s):
            rlo, rhi = mirror_left_to_right(lo[3:], hi[3:])
            # right arm: an independent uniform sample in the mirrored box
            right = rng.uniform(rlo, rhi)
            ok[i], _ = ck.check(upper_vec(x[:3], x[3:], right))
        frac = ok.mean()
        print(f"box iter {it}: feasible {frac:.3f}")
        if frac >= args.feasible_target:
            break
        # shrink the edge bin with the worst failure rate (one per iteration)
        worst, arg = 0.0, None
        for j in range(len(lo)):
            edges = np.linspace(lo[j], hi[j], 11)
            b = np.clip(np.digitize(s[:, j], edges) - 1, 0, 9)
            for side, bi in (("lo", 0), ("hi", 9)):
                sel = b == bi
                if sel.sum() > 10:
                    fail = 1.0 - ok[sel].mean()
                    if fail > worst:
                        worst, arg = fail, (j, side, edges)
        j, side, edges = arg
        if side == "lo":
            lo[j] = edges[1]
        else:
            hi[j] = edges[-2]
    rlo, rhi = mirror_left_to_right(lo[3:], hi[3:])
    box_lo = np.concatenate([lo, rlo])
    box_hi = np.concatenate([hi, rhi])
    m_hats = [e["m_hat"] for e in library]
    out = {
        "generated_by": "scripts/reach_envelope.py (do not edit)",
        "model": os.path.relpath(args.xml, RL),
        "joint_order": [JOINT_NAMES[i] for i in UPPER_IDX],
        "criteria": {"m_hat_min": args.m_hat_min, "no_new_self_contact": True,
                     "static_upper_torque_below_clip": True},
        "support_polygon_m_max_m": round(float(ck.m_max), 4),
        "box_feasible_fraction": round(float(frac), 3),
        "box_lo": [round(float(v), 3) for v in box_lo],
        "box_hi": [round(float(v), 3) for v in box_hi],
        "library_min_m_hat": round(float(min(m_hats)), 3),
        "pelvis_tilt_needed_deg": 0.0,
        "tilt_termination_deg": round(float(np.degrees(0.8)), 1),
        "library": library,
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, width=200)
    print(f"-> {args.out}: {len(library)} poses, box feasible {frac:.2f}, "
          f"min library m_hat {min(m_hats):.2f}")


if __name__ == "__main__":
    main()
