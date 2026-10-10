"""Stage C: ULTRA motion-tracking teacher on MJX (faithful port, no objects).

Physics, PD, DR, pushes, action delay: inherited from G1TeleopEnv (ULTRA
values). Reference: prepared motions (motion_data.MotionLib), one frame per
control step (ULTRA consumes one reference frame per policy step).

Reward = ULTRA humanoid_g1.py compute_humanoid_reward with the object /
interaction / hand-contact factors set to 1 (no objects in this scene):
    rp  = exp(-w_p  * mean_feet |s*ref_foot - foot|^2)          (feet key pos)
    rr  = exp(-w_r  * mean_links |dir_ref - dir|^2)              (link directions)
    rpv = exp(-w_pv * mean |s*ref_key_vel - key_vel|^2)          (key body velocity)
    rb  = rp * rr * rpv * exp(-w_eg1 * mean a^2)
    r   = exp(-20 mean (q - init_dof)^2) * w1(t) + rb * (w2(t) + w3(t))
          w1 = sigmoid((10 - t) / 5), w3 = sigmoid((t - 20) / 2), w2 = 1 - w1 - w3
    r  *= exp(S / 10) for t < 30, exp(S / 15) after; S = sum(ULTRA smooth scale * term)
Termination = ULTRA compute_humanoid_reset: root height < 0.15 (t > 1),
< 0.5 in the first 10 steps, mean feet deviation > 0.5 m after 30 steps.
The robot always starts in the default pose (ULTRA _set_env_state), at the
reference's root xy (x root scale) and heading; the reference starts with
standing frames (prepended by prepare), so the stage-I standing reward and
the blend into tracking behave as in ULTRA.

Observations:
  'state' (129): proprio [ang vel 3, gravity 3, q - q0 29, qd 29, last action
     29] + goals [ref_dof(next) - q 29, ref projected gravity 3, root xy error
     in the heading frame 2, heading error 1, ref root height 1]
     (~ ULTRA student "full_track" local/global goal).
  'privileged_state' (183): teleop privileged (73) + key-body error in the
     heading frame 39 + ref key positions rel. root 39 + ref root velocity 3
     + ref dof 16 frames ahead 29 (ULTRA teacher sees iterations 1 and 16).
"""

import jax
import jax.numpy as jnp
from brax.envs.base import State
from mujoco import mjx

from . import randomize, rewards
from .env import G1TeleopEnv, quat_rotate_inv
from .layout import NU, PRIV_DIM
from .motion_data import MotionLib

TRACK_OBS = 3 + 3 + 29 + 29 + 29 + 29 + 3 + 2 + 1 + 1          # 129


def _yaw(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _yaw_quat(yaw):
    return jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])


def _rot_z(v, yaw):
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


class G1TrackingEnv(G1TeleopEnv):
    def __init__(self, cfg, stage, xml_path=None, include_history=False, student_noise=False,
                 emit_clean_state=False, motion_dir=None):
        kw = {} if xml_path is None else {"xml_path": xml_path}
        super().__init__(cfg, stage, include_history=include_history, student_noise=student_noise,
                         emit_clean_state=emit_clean_state, **kw)
        import os
        from .config import RL_ROOT
        tc = cfg.train.tracking
        self._tc = tc
        lib = MotionLib(os.path.join(RL_ROOT, motion_dir or tc.motion_dir), tc.get("max_motions"))
        if abs(lib.fps - 1.0 / self._ctrl_dt) > 1e-3:
            raise ValueError(f"motions at {lib.fps} Hz, control at {1 / self._ctrl_dt} Hz: re-prepare")
        self._lib = {k: jnp.asarray(getattr(lib, k)) for k in
                     ("root_pos", "root_quat", "dof", "dof_vel", "key_pos", "key_vel", "root_vel")}
        self._len = jnp.asarray(lib.lengths)
        self._n_motions = int(lib.lengths.shape[0])
        m = self._mj_model
        self._key_ids = jnp.array([m.body(b).id for b in tc.key_bodies])
        self._pairs = jnp.array(tc.link_pairs)
        self._feet_k = jnp.array(tc.feet_key_idx)
        self._s = float(tc.ref_root_scale)
        # teleop noise vector covers the first 6 + joint entries of its own layout;
        # rebuild for the tracking layout (ULTRA uniform amplitudes)
        n = self._dr.obs_noise
        amp = jnp.concatenate([jnp.full(3, n.ang_vel), jnp.full(3, n.gravity), jnp.full(29, n.dof_pos),
                               jnp.full(29, n.dof_vel), jnp.zeros(TRACK_OBS - 64)])
        teacher = not (include_history or student_noise)
        self._noise_amp = amp * (float(cfg.train.get("teacher_obs_noise", 0.0)) if teacher else 1.0)

    @property
    def observation_size(self):
        sizes = {"state": TRACK_OBS, "privileged_state": PRIV_DIM + 39 + 39 + 3 + 29}
        if self._emit_clean:
            sizes["state_clean"] = TRACK_OBS
        if self._history_len:
            sizes["history"] = (self._history_len, TRACK_OBS)
        return sizes

    def reward_keys(self):
        return ("track_rp", "track_rr", "track_rpv", "track_standing", "smooth_sum",
                "feet_err_m")

    # ----------------------------------------------------------- reference
    def _ref(self, info, k):
        idx = jnp.minimum(info["t0"] + info["step"] + k, self._len[info["motion"]] - 1)
        g = lambda a: a[info["motion"], idx]  # noqa: E731
        s, o = self._s, info["ref_offset"]
        # ULTRA multiplies reference positions/velocities by `scaling`; the
        # offset/yaw place the reference where the robot started
        rot = lambda v: _rot_z(v, info["ref_yaw"])  # noqa: E731
        return {"dof": g(self._lib["dof"]), "root_quat": g(self._lib["root_quat"]),
                "root_pos": rot(g(self._lib["root_pos"]) * s) + o,
                "key_pos": rot(g(self._lib["key_pos"]) * s) + o,
                "key_vel": rot(g(self._lib["key_vel"]) * s), "root_vel": rot(g(self._lib["root_vel"]) * s)}

    # ----------------------------------------------------------- obs
    def _obs(self, data, info, rng):
        quat = data.qpos[3:7]
        yaw = _yaw(quat)
        proj_g = quat_rotate_inv(quat, jnp.array([0.0, 0.0, -1.0]))
        r1, r16 = self._ref(info, 1), self._ref(info, 16)
        ref_yaw = _yaw(r1["root_quat"]) + info["ref_yaw"]
        ref_g = quat_rotate_inv(r1["root_quat"], jnp.array([0.0, 0.0, -1.0]))
        dxy = _rot_z((r1["root_pos"] - data.qpos[0:3])[None], -yaw)[0, :2]
        clean = jnp.concatenate([
            data.qvel[3:6], proj_g, data.qpos[7:] - self._q0, data.qvel[6:], info["last_action"],
            r1["dof"] - data.qpos[7:], ref_g, dxy, jnp.sin(ref_yaw - yaw)[None],
            r1["root_pos"][2:3]])
        noisy = clean + randomize.sample_uniform_noise(rng, self._noise_amp)
        key = data.xpos[self._key_ids]
        root = data.qpos[0:3]
        v_local = quat_rotate_inv(quat, data.qvel[0:3])
        priv = jnp.concatenate([
            info["friction"][None], info["torso_mass_delta"][None] / 3.0,
            info["torso_com_shift"] / 0.05, info["kp_scale"], info["kd_scale"],
            info["action_delay"][None].astype(jnp.float32), info["gravity_offset"] / 0.1,
            v_local, data.qpos[2:3], info["feet_contact"].astype(jnp.float32),
            _rot_z(r1["key_pos"] - key, -yaw).ravel(),
            _rot_z(r1["key_pos"] - r1["root_pos"], -yaw).ravel(),
            _rot_z(r1["root_vel"][None], -yaw)[0], r16["dof"]])
        out = {"state": noisy, "privileged_state": priv}
        if self._emit_clean:
            out["state_clean"] = clean
        return out

    # ----------------------------------------------------------- reset
    def reset(self, rng):
        k = jax.random.split(rng, 8)
        fields, priv = randomize.sample_model_fields(k[0], self._base, self._dr)
        motion = jax.random.randint(k[1], (), 0, self._n_motions)
        L = self._len[motion]
        rand_start = jax.random.uniform(k[2]) < self._tc.hybrid_init_prob
        t0 = jnp.where(rand_start, jax.random.randint(k[3], (), 0, jnp.maximum(L - 50, 1)), 0)
        ref_q = self._lib["root_quat"][motion, t0]
        ref_p = self._lib["root_pos"][motion, t0] * self._s
        info = {"rng": k[4], "step": jnp.int32(0), "motion": motion, "t0": t0.astype(jnp.int32),
                # the reference is re-anchored so its start sits where the robot stands
                "ref_yaw": jnp.float32(0.0), "ref_offset": jnp.array([-ref_p[0], -ref_p[1], 0.0]),
                "model_fields": fields, **priv, "cmd": jnp.zeros(3),
                "action_buffer": jnp.zeros((2, NU)), "last_action": jnp.zeros(NU),
                "feet_contact": jnp.ones(2, bool),
                "push_next": jax.random.randint(k[5], (), int(self._dr.push.first_push_s[0] / self._ctrl_dt),
                                                int(self._dr.push.first_push_s[1] / self._ctrl_dt))}
        qpos = self._default_qpos.at[3:7].set(_yaw_quat(_yaw(ref_q)))
        qpos = qpos.at[7:].add(jax.random.uniform(k[6], (NU,), minval=-0.03, maxval=0.03))
        model = self._model_with(info)
        data = mjx.forward(model, mjx.make_data(model).replace(qpos=qpos))
        obs = self._obs(data, info, k[7])
        if self._history_len:
            info["obs_hist"] = jnp.tile(obs["state"], (self._history_len, 1))
            obs["history"] = info["obs_hist"]
        metrics = {f"reward/{n}": jnp.float32(0.0) for n in self.reward_keys()}
        metrics["termination"] = jnp.float32(0.0)
        return State(pipeline_state=data, obs=obs, reward=jnp.float32(0.0),
                     done=jnp.float32(0.0), metrics=metrics, info=info)

    # ----------------------------------------------------------- step
    def step(self, state, action):
        tc = self._tc
        info = dict(state.info)
        rng, k_noise, k_push, k_grav = jax.random.split(info["rng"], 4)
        t = info["step"]
        action = jnp.clip(action, -1.0, 1.0)
        buf = jnp.stack([action, info["action_buffer"][0]])
        q_target = jnp.clip(self._q0 + buf[info["action_delay"]] * self._scale, self._q_lo, self._q_hi)
        data0 = state.pipeline_state
        push = t == info["push_next"]
        kick = randomize.sample_push_velocity(k_push, self._dr)
        data0 = data0.replace(qvel=data0.qvel.at[0:2].set(
            jnp.where(push & (self._dr.push.max_vel_xy > 0), kick, data0.qvel[0:2])))
        info["push_next"] = jnp.where(push, t + self._push_every, info["push_next"])
        regrav = (t > 0) & (jnp.mod(t, self._grav_every) == 0)
        info["gravity_offset"] = jnp.where(regrav, randomize.sample_gravity(k_grav, self._dr),
                                           info["gravity_offset"])
        data, tau, dq_sub, dw_sub = self._physics(
            self._model_with(info), data0, q_target, self._kp * info["kp_scale"],
            self._kd * info["kd_scale"])

        info["step"] = t + 1
        ref = self._ref(info, 0)
        key = data.xpos[self._key_ids]
        # key-body linear velocity (world): finite difference over the control step
        key_prev = state.pipeline_state.xpos[self._key_ids]
        key_vel = (key - key_prev) / self._ctrl_dt
        q, qd = data.qpos[7:], data.qvel[6:]
        feet_err = jnp.linalg.norm(ref["key_pos"][self._feet_k] - key[self._feet_k], axis=-1)
        ep = jnp.mean(jnp.sum(jnp.square(ref["key_pos"][self._feet_k] - key[self._feet_k]), -1))

        def dirs(kp):
            v = kp[self._pairs[:, 0]] - kp[self._pairs[:, 1]]
            return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-5)

        er = jnp.mean(jnp.sum(jnp.square(dirs(ref["key_pos"]) - dirs(key)), -1))
        epv = jnp.mean(jnp.square(ref["key_vel"] - key_vel))
        rb = (jnp.exp(-tc.w_p * ep) * jnp.exp(-tc.w_r * er) * jnp.exp(-tc.w_pv * epv)
              * jnp.exp(-tc.w_eg1 * jnp.mean(jnp.square(action))))
        r_stand = jnp.exp(-tc.w_standing_dof * jnp.mean(jnp.square(q - self._q0)))
        tf = t.astype(jnp.float32)
        w1 = jax.nn.sigmoid((10.0 - tf) / 5.0)
        w3 = jax.nn.sigmoid((tf - 20.0) / 2.0)
        w2 = 1.0 - w1 - w3
        r_track = r_stand * w1 + rb * (w2 + w3)

        contact = self._feet_contact(data)
        quat = data.qpos[3:7]
        foot_pos = data.site_xpos[self._foot_sites]
        foot_mat = data.site_xmat[self._foot_sites].reshape(2, 3, 3)
        foot_g = jnp.einsum("fji,j->fi", foot_mat, jnp.array([0.0, 0.0, -1.0]))
        foot_vel = key_vel[self._feet_k]
        knees = data.xpos[self._knees]
        cf = contact.astype(jnp.float32)
        ref_feet_down = jnp.all(ref["key_pos"][self._feet_k, 2] - ref["key_pos"][self._feet_k, 2].min() < 0.04) \
            & jnp.all(ref["key_pos"][self._feet_k, 2] < 0.12)
        rw = self._cfg.rewards
        term = {
            "base_lin_vel": jnp.linalg.norm(data.qvel[0:3]),
            "ang_vel": jnp.sum(jnp.square(data.qvel[3:6])),
            "dof_vel": jnp.sum(jnp.square(qd)),
            "action_rate": jnp.linalg.norm(action - info["last_action"]),
            "torques": jnp.linalg.norm(tau),
            "dof_pos_limits": jnp.sum(jnp.clip(self._q_lo - q, 0.0) + jnp.clip(q - self._q_hi, 0.0)),
            "dof_torque_limits": jnp.sum(jnp.clip(jnp.abs(tau) / self._tau_lim - rw.soft_torque_limit, 0.0)),
            "energy": jnp.linalg.norm(jnp.abs(tau * qd)),
            "delta_dof_vel_all": dq_sub, "delta_ang_vel_all": dw_sub,
            "torques_vel_all": jnp.float32(0.0),
            "feet_orientation": jnp.sum(jnp.linalg.norm(foot_g[:, :2], axis=-1)),
            "foot_slip": jnp.sum(jnp.sqrt(jnp.linalg.norm(foot_vel[:, :2], axis=-1) + 1e-8) * cf),
            "feet_distance": rewards.band_violation(jnp.linalg.norm(foot_pos[0, :2] - foot_pos[1, :2]),
                                                    rw.feet_distance_band),
            "knee_distance": rewards.band_violation(jnp.linalg.norm(knees[0, :2] - knees[1, :2]),
                                                    rw.knee_distance_band),
            "stand_on_feet": (ref_feet_down & ~jnp.all(contact)).astype(jnp.float32),
        }
        h = data.qpos[2]
        fell = ((h < tc.termination_height) & (t > 1)) | \
               ((h < tc.termination_height_init) & (t < tc.init_steps))
        deviated = (jnp.mean(feet_err) > tc.feet_deviation_reset_m) & (t > tc.feet_deviation_grace_steps)
        done = fell | deviated | ~jnp.all(jnp.isfinite(data.qpos))
        sc = tc.smooth_scales
        S = sum(sc[k2] * v for k2, v in term.items()) + sc.termination * done
        div = jnp.where(t < tc.smooth_switch_step, tc.smooth_div[0], tc.smooth_div[1])
        reward = r_track * jnp.exp(S / div)
        reward = jnp.where(jnp.isfinite(reward), reward, 0.0)

        info["rng"] = rng
        info["action_buffer"] = buf
        info["last_action"] = action
        info["feet_contact"] = contact
        obs = self._obs(data, info, k_noise)
        if self._history_len:
            info["obs_hist"] = jnp.roll(info["obs_hist"], -1, axis=0).at[-1].set(obs["state"])
            obs["history"] = info["obs_hist"]
        metrics = dict(state.metrics)
        metrics.update({"reward/track_rp": jnp.exp(-tc.w_p * ep), "reward/track_rr": jnp.exp(-tc.w_r * er),
                        "reward/track_rpv": jnp.exp(-tc.w_pv * epv), "reward/track_standing": r_stand,
                        "reward/smooth_sum": S, "reward/feet_err_m": jnp.mean(feet_err),
                        "termination": done.astype(jnp.float32)})
        del quat
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.float32(reward),
                             done=done.astype(jnp.float32), metrics=metrics, info=info)
