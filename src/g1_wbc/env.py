"""G1 29-DoF teleop whole-body controller env (Stage A stand, Stage B
velocity locomotion + upper-body tracking) on MuJoCo MJX.

Structure ported from the validated AgiBot X1 / H1-2 envs (skill reference
implementations); physics parameters from ULTRA (configs/*.yaml cite each).

Control: the policy outputs 29 joint actions a in [-1, 1];
    q* = clip(default_qpos + a * action_scale, joint limits)
    tau = clip(kp*ms_p*(q* - q) - kd*ms_d*qd, -torque_limit, torque_limit)
recomputed at every 500 Hz physics substep (10 per 50 Hz control step), with
ULTRA's per-joint motor-strength DR (ms_p, ms_d) and 0-1 step action delay.

Observations (dict):
  'state'            (115) the deployable frame, layout.OBS_TERMS. Teacher:
                     noise-free (train.teacher_obs_noise 0); student envs:
                     ULTRA uniform noise.
  'privileged_state' (73)  layout.PRIV_TERMS (teacher/critic only).
  'state_clean'      (115) only with emit_clean_state (distillation labels).
  'history'          (H, 115) only with include_history (student fine-tune).
"""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from . import commands as C
from . import randomize, rewards
from .config import MJX_XML, Config, Stage, base_height_target, stage_commands, stage_domain_rand
from .layout import (HIP_ROLL_YAW_IDX, N_ACT, NU, OBS_DIM, PRIV_DIM, UPPER_IDX, obs_slice,
                     torque_limit)

_FIELD_KEYS = ("body_mass", "body_ipos", "geom_friction")


def quat_rotate_inv(q, v):
    """World vector v in the frame of quaternion q (wxyz)."""
    w, u = q[0], q[1:4]
    return v + 2.0 * jnp.cross(-u, jnp.cross(-u, v) + w * v)


class G1TeleopEnv(Env):
    def __init__(self, cfg: Config, stage: Stage, xml_path: str = MJX_XML,
                 include_history: bool = False, student_noise: bool = False,
                 emit_clean_state: bool = False):
        self._cfg, self._stage = cfg, stage
        self._emit_clean = emit_clean_state
        self._history_len = cfg.train.student.history_len if include_history else 0
        m = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model = m
        self._model = mjx.put_model(m)
        act = cfg.actuators
        self._physics_dt = float(act.control.physics_dt)
        assert abs(m.opt.timestep - self._physics_dt) < 1e-9, "rebuild assets"
        self._decimation = int(act.control.decimation)
        self._ctrl_dt = self._physics_dt * self._decimation
        self._episode_steps = int(cfg.train.env.episode_length_s / self._ctrl_dt)

        self._default_qpos = jnp.asarray(m.key("home").qpos)
        self._q0 = jnp.asarray(act.default_pose, jnp.float32)
        if not np.allclose(m.key("home").qpos[7:], np.asarray(act.default_pose)):
            raise ValueError("home keyframe != actuators.default_pose: rebuild assets")
        self._kp = jnp.asarray(act.kp)
        self._kd = jnp.asarray(act.kd)
        self._scale = jnp.asarray(act.action_scale)
        self._tau_lim = jnp.asarray(torque_limit(act))
        assert np.allclose(m.actuator_ctrlrange[:, 1], np.asarray(torque_limit(act))), \
            "ctrlrange != torque_limit: rebuild assets"
        self._q_lo = jnp.asarray(m.jnt_range[1:, 0])
        self._q_hi = jnp.asarray(m.jnt_range[1:, 1])
        self._height_target = base_height_target(cfg)

        self._torso_id = m.body("torso_link").id
        self._floor = m.geom("floor").id
        self._sole = jnp.array([[m.geom(f"{s}_sole_{k}").id for k in range(4)]
                                for s in ("left", "right")])
        self._foot_sites = jnp.array([m.site("left_foot").id, m.site("right_foot").id])
        self._knees = jnp.array([m.body("left_knee_link").id, m.body("right_knee_link").id])
        fric_ids = jnp.concatenate([jnp.array([self._floor]), self._sole.ravel()])
        self._base = randomize.ModelBase(
            body_mass=jnp.asarray(m.body_mass), body_ipos=jnp.asarray(m.body_ipos),
            geom_friction=jnp.asarray(m.geom_friction), torso_id=self._torso_id,
            friction_geom_ids=fric_ids)
        self._gravity0 = jnp.asarray(m.opt.gravity)

        self._dr = stage_domain_rand(cfg.domain_rand, stage)
        self._cmd_ranges = stage_commands(cfg.commands, stage)
        self._bank = C.UpperBodyBank(cfg.commands, act.default_pose,
                                     m.jnt_range[1:, 0], m.jnt_range[1:, 1])
        self._push_every = int(round(self._dr.push.interval_s / self._ctrl_dt))
        self._grav_every = int(round(self._dr.gravity.interval_s / self._ctrl_dt))

        n = self._dr.obs_noise
        amp = np.zeros(OBS_DIM, np.float32)
        amp[obs_slice("base_ang_vel")] = n.ang_vel
        amp[obs_slice("projected_gravity")] = n.gravity
        amp[obs_slice("joint_pos_rel")] = n.dof_pos
        amp[obs_slice("joint_vel")] = n.dof_vel
        if not (include_history or student_noise):
            amp *= float(cfg.train.get("teacher_obs_noise", 0.0))
        self._noise_amp = jnp.asarray(amp)
        self._priv_push = bool(cfg.train.get("privileged_push", False))
        assert not self._priv_push, "push kicks are not inferable from history (L20)"

    # ------------------------------------------------------------ brax API
    @property
    def observation_size(self):
        sizes = {"state": OBS_DIM, "privileged_state": PRIV_DIM}
        if self._emit_clean:
            sizes["state_clean"] = OBS_DIM
        if self._history_len:
            sizes["history"] = (self._history_len, OBS_DIM)
        return sizes

    @property
    def action_size(self):
        return N_ACT

    @property
    def backend(self):
        return "mjx"

    @property
    def dt(self):
        return self._ctrl_dt

    @property
    def episode_steps(self):
        return self._episode_steps

    @property
    def stage(self):
        return self._stage

    def reward_keys(self):
        """Metric names DERIVED from the weights (skill M4)."""
        return tuple(k for k in self._cfg.rewards.weights if k != "termination")

    # ------------------------------------------------------------ helpers
    def _model_with(self, info):
        f = info["model_fields"]
        model = self._model.replace(**{k: f[k] for k in _FIELD_KEYS})
        return model.replace(opt=model.opt.replace(gravity=self._gravity0 + info["gravity_offset"]))

    def _physics(self, model, data, q_target, kp, kd):
        qd0 = data.qvel[6:]
        w0 = data.qvel[3:6]

        def substep(carry, _):
            d, dq_acc, dw_acc = carry
            tau = kp * (q_target - d.qpos[7:]) - kd * d.qvel[6:]
            tau = jnp.clip(tau, -self._tau_lim, self._tau_lim)
            d = mjx.step(model, d.replace(ctrl=tau))
            dq_acc = dq_acc + jnp.sum(jnp.square(d.qvel[6:] - qd0))
            dw_acc = dw_acc + jnp.sum(jnp.square(d.qvel[3:6] - w0))
            return (d, dq_acc, dw_acc), tau

        (data, dq, dw), taus = jax.lax.scan(substep, (data, 0.0, 0.0), None, self._decimation)
        return data, taus[-1], dq, dw

    def _feet_contact(self, data):
        con = getattr(data, "_impl", data).contact     # mjx 3.10: data._impl.contact
        g = con.geom
        pen = con.dist < 0.0

        def touching(ids):
            return jnp.any((jnp.isin(g[:, 0], ids) | jnp.isin(g[:, 1], ids)) & pen)

        return jax.vmap(touching)(self._sole)

    def _obs(self, data, info, rng):
        quat = data.qpos[3:7]
        proj_g = quat_rotate_inv(quat, jnp.array([0.0, 0.0, -1.0]))
        clock, _ = C.gait_clock(info["step"], self._ctrl_dt, self._cfg.commands.gait)
        clean = jnp.concatenate([
            data.qvel[3:6], proj_g, info["cmd"], info["cmd_h"][None],
            info["ub_target"] - self._q0[jnp.array(UPPER_IDX)],
            data.qpos[7:] - self._q0, data.qvel[6:], info["last_action"], clock])
        noisy = clean + randomize.sample_uniform_noise(rng, self._noise_amp)
        v_local = quat_rotate_inv(quat, data.qvel[0:3])
        priv = jnp.concatenate([
            info["friction"][None], info["torso_mass_delta"][None] / 3.0,
            info["torso_com_shift"] / 0.05, info["kp_scale"], info["kd_scale"],
            info["action_delay"][None].astype(jnp.float32),
            info["gravity_offset"] / 0.1, v_local, data.qpos[2:3],
            info["feet_contact"].astype(jnp.float32)])
        out = {"state": noisy, "privileged_state": priv}
        if self._emit_clean:
            out["state_clean"] = clean
        return out

    # ------------------------------------------------------------ reset
    def reset(self, rng):
        st, cmds = self._stage, self._cfg.commands
        k = jax.random.split(rng, 10)
        fields, priv = randomize.sample_model_fields(k[0], self._base, self._dr)
        mode1, mode2, switch = C.sample_modes(k[1], st, cmds.modes, self._ctrl_dt)
        info = {
            "rng": k[2], "step": jnp.int32(0),
            "mode1": mode1, "mode2": mode2, "switch_step": switch,
            "cmd_raw": C.sample_cmd_vel(k[3], self._cmd_ranges, cmds.cmd_vel.zero_below,
                                        cmds.cmd_vel.p_zero_axis),
            "cmd_next": jnp.int32(int(cmds.cmd_vel.resample_s[1] / self._ctrl_dt)),
            "cmd": jnp.zeros(3),
            # optional base-height command (default-off: range [0, 0])
            "cmd_h": jax.random.uniform(k[8], minval=cmds.base_height.offset_range[0],
                                        maxval=cmds.base_height.offset_range[1]),
            "model_fields": fields, **priv,
            "action_buffer": jnp.zeros((2, N_ACT)),
            "last_action": jnp.zeros(N_ACT),
            "last_base_vel": jnp.zeros(3),
            "last_foot_pos": jnp.zeros((2, 3)),
            "feet_air_time": jnp.zeros(2),
            "feet_contact": jnp.ones(2, bool),
            "push_next": jax.random.randint(k[4], (), int(self._dr.push.first_push_s[0] / self._ctrl_dt),
                                            int(self._dr.push.first_push_s[1] / self._ctrl_dt)),
        }
        info.update(C.init_upper(k[5], self._bank, cmds.upper_body,
                                 C.mode_at(info, 0), st.arm_scale, self._ctrl_dt))
        info["cmd"] = jnp.where(C.mode_at(info, 0) == C.LOCO, info["cmd_raw"], jnp.zeros(3))

        qpos = self._default_qpos
        qpos = qpos.at[7:].add(jax.random.uniform(k[6], (NU,), minval=-0.05, maxval=0.05))
        qpos = qpos.at[7:].set(jnp.clip(qpos[7:], self._q_lo, self._q_hi))
        model = self._model_with(info)
        data = mjx.make_data(model).replace(qpos=qpos)
        data = mjx.forward(model, data)
        info["last_foot_pos"] = data.site_xpos[self._foot_sites]
        obs = self._obs(data, info, k[7])
        if self._history_len:
            hist = jnp.tile(obs["state"], (self._history_len, 1))
            info["obs_hist"] = hist
            obs["history"] = hist
        metrics = {f"reward/{n}": jnp.float32(0.0) for n in self.reward_keys()}
        metrics.update(termination=jnp.float32(0.0), upper_err=jnp.float32(0.0),
                       mode_manip=jnp.float32(0.0))
        return State(pipeline_state=data, obs=obs, reward=jnp.float32(0.0),
                     done=jnp.float32(0.0), metrics=metrics, info=info)

    # ------------------------------------------------------------ step
    def step(self, state, action):
        cfg, st = self._cfg, self._stage
        info = dict(state.info)
        rng, k_noise, k_push, k_grav, k_ub, k_cmd = jax.random.split(info["rng"], 6)
        step_i = info["step"]
        info = C.advance_upper(info, step_i, k_ub, self._bank, cfg.commands.upper_body,
                               st.arm_scale, self._ctrl_dt)
        info = C.advance_cmd(info, step_i, k_cmd, self._cmd_ranges, cfg.commands.cmd_vel,
                             self._ctrl_dt)

        action = jnp.clip(action, -1.0, 1.0)
        buf = jnp.stack([action, info["action_buffer"][0]])
        applied = buf[info["action_delay"]]
        q_target = jnp.clip(self._q0 + applied * self._scale, self._q_lo, self._q_hi)

        # ULTRA push: base xy velocity SET to U(-v, v) every interval_s
        data0 = state.pipeline_state
        push = step_i == info["push_next"]
        kick = randomize.sample_push_velocity(k_push, self._dr)
        qvel = data0.qvel.at[0:2].set(jnp.where(push & (self._dr.push.max_vel_xy > 0),
                                                kick, data0.qvel[0:2]))
        data0 = data0.replace(qvel=qvel)
        info["push_next"] = jnp.where(push, step_i + self._push_every, info["push_next"])
        # ULTRA gravity DR: resampled every interval_s
        regrav = (step_i > 0) & (jnp.mod(step_i, self._grav_every) == 0)
        info["gravity_offset"] = jnp.where(regrav, randomize.sample_gravity(k_grav, self._dr),
                                           info["gravity_offset"])

        model = self._model_with(info)
        data, tau, dq_sub, dw_sub = self._physics(
            model, data0, q_target, self._kp * info["kp_scale"], self._kd * info["kd_scale"])

        quat = data.qpos[3:7]
        proj_g = quat_rotate_inv(quat, jnp.array([0.0, 0.0, -1.0]))
        v_local = quat_rotate_inv(quat, data.qvel[0:3])
        w_local = data.qvel[3:6]
        contact = self._feet_contact(data)
        first_contact = contact & (info["feet_air_time"] > 0.0)
        foot_pos = data.site_xpos[self._foot_sites]
        foot_vel = (foot_pos - info["last_foot_pos"]) / self._ctrl_dt
        foot_mat = data.site_xmat[self._foot_sites].reshape(2, 3, 3)
        foot_g = jnp.einsum("fji,j->fi", foot_mat, jnp.array([0.0, 0.0, -1.0]))
        knees = data.xpos[self._knees]
        _, stance = C.gait_clock(step_i, self._ctrl_dt, cfg.commands.gait)
        q = data.qpos[7:]

        x = rewards.RewardInputs(
            cmd=info["cmd"], lin_vel_local=v_local, ang_vel_local=w_local,
            proj_gravity=proj_g, base_height=data.qpos[2],
            base_acc=(data.qvel[0:3] - info["last_base_vel"]) / self._ctrl_dt,
            upper_q=q[jnp.array(UPPER_IDX)], upper_target=info["ub_target"],
            q=q, qd=data.qvel[6:], q_lo=self._q_lo, q_hi=self._q_hi,
            tau=tau, tau_limit=self._tau_lim, action=action, prev_action=info["last_action"],
            dqd_sub=dq_sub, dw_sub=dw_sub, feet_contact=contact, first_contact=first_contact,
            feet_air_time=info["feet_air_time"], feet_height=foot_pos[:, 2], feet_vel=foot_vel,
            feet_gravity_xy=foot_g[:, :2],
            feet_lateral=jnp.linalg.norm(foot_pos[0, :2] - foot_pos[1, :2]),
            knee_lateral=jnp.linalg.norm(knees[0, :2] - knees[1, :2]),
            stance_clock=stance, hip_roll_yaw=q[jnp.array(HIP_ROLL_YAW_IDX)])
        reward, terms = rewards.compute_reward(x, cfg.rewards,
                                               self._height_target + info["cmd_h"])

        term = cfg.rewards.termination
        done = (data.qpos[2] < term.base_height_min) | rewards.tilt_exceeded(
            proj_g, term.max_tilt, term.max_pitch_forward)
        done = done | ~jnp.all(jnp.isfinite(data.qpos))
        reward = reward + cfg.rewards.weights.termination * done
        reward = jnp.where(jnp.isfinite(reward), reward, 0.0)

        info["rng"] = rng
        info["step"] = step_i + 1
        info["action_buffer"] = buf
        info["last_action"] = action
        info["last_base_vel"] = data.qvel[0:3]
        info["last_foot_pos"] = foot_pos
        info["feet_air_time"] = jnp.where(contact, 0.0, info["feet_air_time"] + self._ctrl_dt)
        info["feet_contact"] = contact

        obs = self._obs(data, info, k_noise)
        if self._history_len:
            hist = jnp.roll(info["obs_hist"], -1, axis=0).at[-1].set(obs["state"])
            info["obs_hist"] = hist
            obs["history"] = hist
        metrics = dict(state.metrics)
        for n, v in terms.items():
            metrics[f"reward/{n}"] = v
        metrics["termination"] = done.astype(jnp.float32)
        metrics["upper_err"] = jnp.sqrt(jnp.mean(jnp.square(x.upper_q - x.upper_target)))
        metrics["mode_manip"] = (C.mode_at(info, step_i) == C.MANIP).astype(jnp.float32)
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.float32(reward),
                             done=done.astype(jnp.float32), metrics=metrics, info=info)
