"""Generated assets vs the contract (skill pre-flight C)."""
import json
import os

import mujoco
import numpy as np
import pytest

from g1_wbc.config import ASSETS_DIR, EVAL_XML, MJX_XML, load_config
from g1_wbc.layout import JOINT_NAMES, torque_limit

cfg = load_config()
HANDS_XML = os.path.join(ASSETS_DIR, "g1_eval_hands.xml")


@pytest.fixture(scope="module", params=[MJX_XML, EVAL_XML, HANDS_XML])
def model(request):
    return mujoco.MjModel.from_xml_path(request.param)


def test_body_joints_by_name_and_actuators(model):
    for i, n in enumerate(JOINT_NAMES):
        a = model.actuator(n)
        assert model.actuator_trnid[a.id, 0] == model.joint(n).id
        assert np.isclose(model.actuator_ctrlrange[a.id, 1], torque_limit(cfg.actuators)[i])
        assert np.isclose(model.dof_armature[model.jnt_dofadr[model.joint(n).id]],
                          cfg.actuators.armature[i])


def test_training_model_qpos_order_is_contract_order():
    m = mujoco.MjModel.from_xml_path(MJX_XML)
    assert m.nu == 29 and m.nq == 36
    for i, n in enumerate(JOINT_NAMES):
        assert m.jnt_qposadr[m.joint(n).id] == 7 + i          # M15
        assert m.actuator(i).name == n


def test_home_keyframe_is_default_pose(model):
    q = model.key("home").qpos
    for i, n in enumerate(JOINT_NAMES):
        assert np.isclose(q[model.jnt_qposadr[model.joint(n).id]], cfg.actuators.default_pose[i])


def _soles(m):
    out = []
    for g in range(m.ngeom):
        n = m.geom(g).name or ""
        if n.startswith(("left_sole_", "right_sole_")):
            out.append((n, m.body(m.geom_bodyid[g]).name, tuple(np.round(m.geom_pos[g], 5)),
                        float(m.geom_size[g][0])))
    return sorted(out)


def test_sole_geometry_identical_and_only_foot_floor_contact():
    ms = [mujoco.MjModel.from_xml_path(p) for p in (MJX_XML, EVAL_XML, HANDS_XML)]
    assert _soles(ms[0]) == _soles(ms[1]) == _soles(ms[2])   # M14
    assert len(_soles(ms[0])) == 8
    m = ms[0]    # training model: only the soles collide
    coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    assert sorted(m.geom(g).name for g in coll) == sorted([s[0] for s in _soles(m)] + ["floor"])


def test_feet_on_floor_at_home():
    m = mujoco.MjModel.from_xml_path(MJX_XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    z = [d.geom_xpos[m.geom(s[0]).id][2] - s[3] for s in _soles(m)]
    assert np.allclose(z, 0.0, atol=2e-3)
    info = json.load(open(os.path.join(ASSETS_DIR, "model_info.json")))
    assert abs(info["train"]["info"]["pelvis_height_home_m"] - d.qpos[2]) < 1e-6


def test_ankle_roll_mirror_orientation():
    """The mirror check never moves ankle roll (skill C): +r left / -r right
    must tilt the soles mirror-symmetrically."""
    m = mujoco.MjModel.from_xml_path(MJX_XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    d.qpos[7 + 5], d.qpos[7 + 11] = 0.15, -0.15
    mujoco.mj_forward(m, d)
    zl = d.site_xmat[m.site("left_foot").id].reshape(3, 3)[:, 2]
    zr = d.site_xmat[m.site("right_foot").id].reshape(3, 3)[:, 2]
    assert np.allclose(zl * [1, -1, 1], zr, atol=1e-6)


def test_zero_action_hold_falls_after_standing():
    """Correct physics: a PD hold at q0 stands for a while, then falls (skill C)."""
    m = mujoco.MjModel.from_xml_path(MJX_XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    kp, kd = np.asarray(cfg.actuators.kp), np.asarray(cfg.actuators.kd)
    q0 = np.asarray(cfg.actuators.default_pose)
    t_fall = None
    for k in range(2000):
        d.ctrl[:] = kp * (q0 - d.qpos[7:]) - kd * d.qvel[6:]
        mujoco.mj_step(m, d)
        if d.qpos[2] < 0.6:
            t_fall = k * m.opt.timestep
            break
    assert t_fall is not None and 0.5 < t_fall < 3.0
