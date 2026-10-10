"""Plain-MuJoCo sim2sim evaluator (CPU) on the full-fidelity eval models.

The ONNX path consumes ONLY policy.onnx + policy.json, exactly as g1_controller
must: joints are addressed BY NAME (the BrainCo full-hands model interleaves
finger joints in qpos), the observation is assembled from the sidecar's "obs"
list by name, the history ring buffer follows sidecar "history", and PD uses
the sidecar kp/kd/torque_limit at sim_dt. It is therefore also the reference
implementation of the controller-side contract.

Termination = rewards.tilt_exceeded + base height, the SAME function as
training (skill M1). Scenarios are built from configs (battery.py), inside the
training envelope (M2).
"""

import dataclasses
import json
import os

import mujoco
import numpy as np

from . import rewards
from .config import EVAL_XML, load_config


def quat_rotate_inv_np(q, v):
    w, u = q[0], q[1:4]
    return v + 2.0 * np.cross(-u, np.cross(-u, v) + w * v)


def min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


@dataclasses.dataclass
class Scenario:
    name: str
    duration_s: float = 20.0
    cmd: list = dataclasses.field(default_factory=list)       # [(t, [vx, vy, wz])] held
    upper: list = dataclasses.field(default_factory=list)     # [(t_start, t_end, pose17)] min-jerk moves
    kicks: list = dataclasses.field(default_factory=list)     # [(t, [vx, vy])] ULTRA velocity kicks
    base_height: float = 0.0


@dataclasses.dataclass
class Result:
    survived: bool
    t_survived: float
    vel_err: float          # mean |v_xy - cmd_xy| + |wz - cmd_wz| while commanded
    upper_rms: float        # mean RMS upper-body joint error
    tilt_deg_max: float
    torque_sat_frac: float  # fraction of joint-steps at the torque limit


class OnnxPolicy:
    """Deployment artefact: policy.onnx + policy.json."""

    def __init__(self, export_dir):
        import onnxruntime as ort
        with open(os.path.join(export_dir, "policy.json")) as f:
            self.sc = json.load(f)
        self.sess = ort.InferenceSession(os.path.join(export_dir, self.sc["onnx"]["file"]),
                                         providers=["CPUExecutionProvider"])
        self.H = int(self.sc["history_len"])
        self.hist = None

    def reset(self):
        self.hist = None

    def __call__(self, frame, priv=None):
        f = frame.astype(np.float32)
        if self.hist is None:                       # sidecar history.reset_fill = first_obs
            self.hist = np.repeat(f[None], self.H, axis=0)
        else:
            self.hist = np.concatenate([self.hist[1:], f[None]], axis=0)
        flat = self.hist.reshape(1, -1)          # (1, H * D), oldest first
        return self.sess.run(None, {self.sc["onnx"]["input"]: flat})[0][0]


class TeacherPolicy:
    """Privileged JAX teacher (battery before distillation); privileged obs are
    built from the simulator with NOMINAL DR latents."""

    needs_priv = True

    def __init__(self, ckpt, cfg=None):
        import jax
        import jax.numpy as jnp
        from .layout import N_ACT
        from .networks import deterministic_action, teacher_policy_module
        from .ppo_teacher import load_checkpoint, rehydrate_normalizer
        cfg = cfg or load_config()
        p, _ = load_checkpoint(ckpt)
        norm = rehydrate_normalizer(p[0])
        mod = teacher_policy_module(cfg, N_ACT)
        sm, ss = jnp.asarray(norm.mean["state"]), jnp.asarray(norm.std["state"])
        pm, ps = jnp.asarray(norm.mean["privileged_state"]), jnp.asarray(norm.std["privileged_state"])
        prm = p[1]
        self._f = jax.jit(lambda s, q: deterministic_action(
            mod.apply(prm, jnp.concatenate([(s - sm) / ss, (q - pm) / ps], -1))))

    def reset(self):
        pass

    def __call__(self, frame, priv):
        return np.asarray(self._f(frame, priv))


class StudentCkptPolicy:
    """Distilled student checkpoint (JAX) with the deployment history rule."""

    def __init__(self, ckpt, cfg=None):
        from .distill_student import StudentJax
        cfg = cfg or load_config()
        self.sj = StudentJax(ckpt, cfg)
        self.H = int(cfg.train.student.history_len)
        self.hist = None

    def reset(self):
        self.hist = None

    def __call__(self, frame, priv=None):
        f = frame.astype(np.float32)
        self.hist = np.repeat(f[None], self.H, 0) if self.hist is None else \
            np.concatenate([self.hist[1:], f[None]], 0)
        return self.sj(self.hist)


class Sim2Sim:
    def __init__(self, sidecar: dict, xml=EVAL_XML, cfg=None):
        self.sc, self.cfg = sidecar, cfg or load_config()
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        names = sidecar["joint_names"]
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in names])
        self.vadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in names])
        self.aid = np.array([self.m.actuator(n).id for n in names])
        self.lo = np.array([l[0] for l in sidecar["joint_limits"]])
        self.hi = np.array([l[1] for l in sidecar["joint_limits"]])
        self.q0 = np.asarray(sidecar["default_qpos"])
        self.scale = np.asarray(sidecar["action_scale"])
        self.kp, self.kd = np.asarray(sidecar["kp"]), np.asarray(sidecar["kd"])
        self.tau_lim = np.asarray(sidecar["torque_limit"])
        self.upper_idx = np.asarray(sidecar["upper_body_joint_indices"])
        self.dec = int(round(sidecar["control_dt"] / sidecar["sim_dt"]))
        assert abs(self.m.opt.timestep - sidecar["sim_dt"]) < 1e-9
        body = set(self.aid.tolist())
        self.extra = [(self.m.actuator_trnid[i, 0], i) for i in range(self.m.nu) if i not in body]
        self.gait = sidecar["gait"]
        term = self.cfg.rewards.termination
        self.h_min, self.tilt, self.fwd = term.base_height_min, term.max_tilt, term.max_pitch_forward
        self.torso = self.m.body("torso_link").id

    # ---------------------------------------------------------- obs from sidecar
    def frame(self, t, cmd, ub_target, cmd_h, last_action):
        d = self.d
        quat = d.qpos[3:7]
        parts = {
            "base_ang_vel": d.qvel[3:6],
            "projected_gravity": quat_rotate_inv_np(quat, np.array([0.0, 0.0, -1.0])),
            "cmd_vel": cmd,
            "cmd_base_height": np.array([cmd_h]),
            "upper_body_target_rel": ub_target - self.q0[self.upper_idx],
            "joint_pos_rel": d.qpos[self.qadr] - self.q0,
            "joint_vel": d.qvel[self.vadr],
            "last_action": last_action,
            "gait_phase": np.array([np.sin(2 * np.pi * self._phase(t)),
                                    np.cos(2 * np.pi * self._phase(t))]),
        }
        return np.concatenate([np.asarray(parts[o["name"]], float) * o["scale"]
                               for o in self.sc["obs"]]).astype(np.float32)

    def _phase(self, t):
        return (t % self.gait["period_s"]) / self.gait["period_s"]

    def priv(self):
        """Nominal-DR privileged vector (layout.PRIV_TERMS order)."""
        d = self.d
        v = quat_rotate_inv_np(d.qpos[3:7], d.qvel[0:3])
        contact = np.zeros(2)
        for i in range(d.ncon):
            c = d.contact[i]
            for g in (c.geom1, c.geom2):
                n = self.m.geom(g).name or ""
                if n.startswith("left_sole_"):
                    contact[0] = 1
                elif n.startswith("right_sole_"):
                    contact[1] = 1
        return np.concatenate([[1.0], [0.0], np.zeros(3), np.ones(29), np.ones(29), [0.0],
                               np.zeros(3), v, [d.qpos[2]], contact]).astype(np.float32)

    # ---------------------------------------------------------- episode
    def run(self, sc: Scenario, policy, seed=0, record=False):
        m, d = self.m, self.d
        rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        d.qpos[self.qadr] += rng.uniform(-0.03, 0.03, len(self.qadr))
        mujoco.mj_forward(m, d)
        policy.reset()
        ctrl_dt = self.sc["control_dt"]
        n = int(sc.duration_s / ctrl_dt)
        last_a = np.zeros(len(self.q0), np.float32)
        ub_def = self.q0[self.upper_idx].copy()
        cmd = np.zeros(3)
        verr, uerr, tilt, sat, trace = [], [], [], [], []
        kicks = sorted(sc.kicks, key=lambda k: k[0])
        ki = 0
        for k in range(n):
            t = k * ctrl_dt
            for tc, c in sc.cmd:
                if t >= tc:
                    cmd = np.asarray(c, float)
            ub = ub_def.copy()          # sequential min-jerk moves, each from the previous pose
            for (t0, t1, pose) in sc.upper:
                if t >= t0:
                    ub = ub + min_jerk((t - t0) / max(t1 - t0, 1e-6)) * (np.asarray(pose) - ub)
            if ki < len(kicks) and t >= kicks[ki][0]:
                d.qvel[0:2] = kicks[ki][1]
                ki += 1
            frame = self.frame(t, cmd, ub, sc.base_height, last_a)
            a = np.clip(np.asarray(policy(frame, self.priv() if getattr(policy, "needs_priv", False) else None),
                                   np.float64), -1, 1)
            last_a = a.astype(np.float32)
            qt = np.clip(self.q0 + a * self.scale, self.lo, self.hi)
            for _ in range(self.dec):
                tau = self.kp * (qt - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr]
                tau = np.clip(tau, -self.tau_lim, self.tau_lim)
                d.ctrl[self.aid] = tau
                for j, ai in self.extra:        # finger joints PD-held open
                    d.ctrl[ai] = np.clip(-2.0 * d.qpos[m.jnt_qposadr[j]] - 0.05 * d.qvel[m.jnt_dofadr[j]],
                                         *m.actuator_ctrlrange[ai])
                mujoco.mj_step(m, d)
            sat.append(np.mean(np.abs(tau) >= 0.999 * self.tau_lim))
            g = quat_rotate_inv_np(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
            v = quat_rotate_inv_np(d.qpos[3:7], d.qvel[0:3])
            tilt.append(np.degrees(np.arcsin(min(1.0, np.linalg.norm(g[:2])))))
            if np.linalg.norm(cmd) > 0.05:
                verr.append(np.linalg.norm(v[:2] - cmd[:2]) + abs(d.qvel[5] - cmd[2]))
            uerr.append(np.sqrt(np.mean((d.qpos[self.qadr][self.upper_idx] - ub) ** 2)))
            if record:
                trace.append(d.qpos.copy())
            if d.qpos[2] < self.h_min or rewards.tilt_exceeded(g, self.tilt, self.fwd, np):
                r = Result(False, t, _m(verr), _m(uerr), max(tilt), float(np.mean(sat)))
                return (r, np.array(trace)) if record else r
        r = Result(True, sc.duration_s, _m(verr), _m(uerr), max(tilt), float(np.mean(sat)))
        return (r, np.array(trace)) if record else r


def _m(x):
    return float(np.mean(x)) if x else 0.0


def wilson(k, n, z=1.96):
    """Wilson score interval (skill theory 12)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))
