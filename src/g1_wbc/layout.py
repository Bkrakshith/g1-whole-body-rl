"""G1 29-DoF joint layout (docs/interfaces.md section 2), read once from
configs/actuators.yaml so the env, evaluator, export and tests cannot
disagree."""

import os

import yaml

JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_ACT = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "actuators.yaml")
with open(_ACT) as _f:
    _L = yaml.safe_load(_f)["layout"]

NU = int(_L["n_joints"])                         # 29
assert NU == len(JOINT_NAMES)
LEG_IDX = tuple(_L["leg_idx"])
WAIST_IDX = tuple(_L["waist_idx"])
ARM_IDX = tuple(_L["arm_idx"])
UPPER_IDX = tuple(_L["upper_idx"])               # 17 = waist 3 + arms 14
N_UPPER = len(UPPER_IDX)
KNEE_IDX = tuple(_L["knee_idx"])
HIP_ROLL_YAW_IDX = tuple(_L["hip_roll_yaw_idx"])
N_ACT = NU                                        # policy outputs all 29 joints
UPPER_NAMES = tuple(JOINT_NAMES[i] for i in UPPER_IDX)


def effort(act) -> list:
    """Per-joint effort [Nm] per actuators.yaml effort_source."""
    return list(act[f"effort_{act['effort_source']}"])


def torque_limit(act) -> list:
    """PD torque clip = effort x torque_limit_fraction (ULTRA 0.8)."""
    f = float(act["torque_limit_fraction"])
    return [f * e for e in effort(act)]


# ---------------------------------------------------------------- observation
# ONE deployable observation frame (Stage A/B teacher 'state', student history
# frames, policy.json "obs"). Order and meaning are the contract with
# g1_controller (every definition is also written into policy.json
# "obs_definitions"). Values are RAW (scale 1.0): normalisation is baked into
# the ONNX graph. The ONNX input is the FLAT concatenation of history_len
# frames, oldest first (1, H * OBS_DIM), as g1_controller feeds it.
OBS_TERMS = (
    ("base_ang_vel", 3),          # pelvis IMU gyro, rad/s, pelvis frame
    ("projected_gravity", 3),     # unit gravity vector (0,0,-1) in pelvis frame
    ("cmd_vel", 3),               # /g1/cmd_vel [vx m/s, vy m/s, wz rad/s] (0 when absent)
    ("cmd_base_height", 1),       # pelvis height command - home height (m); OPTIONAL,
                                  # default-off: always 0 in training so far (sidecar
                                  # command_details.base_height.trained_range_offset_m [0, 0])
    ("upper_body_target_rel", 17),  # /g1/upper_body_target - default_qpos[12:29] (0 when absent)
    ("joint_pos_rel", 29),        # q - default_qpos, all 29 joints, contract order
    ("joint_vel", 29),            # qd, rad/s, all 29 joints
    ("last_action", 29),          # previous policy output (before scaling), in [-1, 1]
    ("gait_phase", 2),            # [sin 2 pi phase, cos 2 pi phase], phase = (t mod 0.8)/0.8
)
# Names/semantics = ros2_ws/src/hospital-sim-ros2/g1_controller/g1_controller/policy.py terms
# (joint terms carry "joints": "all", gait_phase carries "period" in the sidecar).
OBS_DIM = sum(n for _, n in OBS_TERMS)      # 116


def obs_slice(name):
    i = 0
    for n, d in OBS_TERMS:
        if n == name:
            return slice(i, i + d)
        i += d
    raise KeyError(name)


# Privileged (teacher / critic only): every randomised latent (skill L14) that
# persists over the episode (L20; the push kick is NOT here) + base state.
PRIV_TERMS = (
    ("friction", 1), ("torso_mass_delta", 1), ("torso_com_shift", 3),
    ("kp_scale", 29), ("kd_scale", 29), ("action_delay", 1), ("gravity_offset", 3),
    ("base_lin_vel", 3), ("base_height", 1), ("feet_contact", 2),
)
PRIV_DIM = sum(n for _, n in PRIV_TERMS)    # 73
