"""Generate an ULTRA-compatible MJCF of our G1 29-DoF rev 1.0 + BrainCo Revo2 robot.

ULTRA's released teacher builds its observation from 39 named bodies of its own G1 model (rubber hands, head,
IMU, camera and lidar frames as separate bodies) and drives exactly 29 torque actuators. Our model fuses those
fixed frames into their parents and adds 12 actuated BrainCo finger joints. This script keeps OUR dynamics
(masses, inertias, BrainCo hands and finger geometry, joint limits, effort limits) and only adds what the
policy's observation needs:

  * massless marker bodies with ULTRA's names, at ULTRA's frames (read from ULTRA's g1_29dof.xml);
  * fingers without actuators: each actuated finger joint holds a pose with a passive spring (open by default,
    --finger-curl 0..1 to pre-curl), so the model has exactly the 29 body motors, in contract order;
  * no keyframe (ULTRA appends the object's free joint at run time);
  * ULTRA's ground plane and solver options (simulator settings, not robot properties): our robot file keeps its
    floor in a separate scene file, ULTRA's carries its own;
  * body motor ctrlrange from the chosen effort set: `model` (35 Nm ankles / waist roll-pitch, the user's choice)
    or `ultra` (50 Nm, what ULTRA trained with).

    ultra_eval/run.sh build_ultra_robot.py       # writes ultra_eval/assets/g1_brainco_ultra_{model,ultra}.xml
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)                  # g1-whole-body-rl
REPO = os.path.dirname(RL)                  # hospital-sim master checkout (sibling repos)
SRC = os.path.join(REPO, "g1-brainco-description", "g1_29dof_brainco", "mjcf", "g1_29dof_brainco.xml")
SPEC = os.path.join(REPO, "g1-brainco-description", "g1_29dof_brainco", "robot_spec.json")
ULTRA_G1 = os.path.join(RL, "third_party", "ULTRA", "ultra", "data", "assets", "g1", "g1_29dof.xml")
OUT = os.path.join(HERE, "assets")

MARKERS = ["pelvis_contour_link", "imu_in_pelvis", "logo_link", "head_link", "imu_in_torso", "d435_link",
           "mid360_link", "left_rubber_hand", "right_rubber_hand"]
FINGER_SPRING = 0.5     # N·m/rad: holds the pose, yields on contact (Revo2 drives are back-drivable)


def ultra_marker_frames():
    m = mujoco.MjModel.from_xml_path(ULTRA_G1)
    frames = {}
    for n in MARKERS:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        parent = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b])
        frames[n] = (parent, m.body_pos[b].tolist(), m.body_quat[b].tolist())
    return frames


def build(effort_set, finger_curl):
    tree = ET.parse(SRC)
    root = tree.getroot()
    comp = root.find("compiler")
    # mesh files: resolve against the source file (+ its meshdir), then write ONE meshdir relative to the
    # output file and the file names relative to it, so the checkout stays relocatable
    src_meshdir = os.path.join(os.path.dirname(SRC), comp.get("meshdir") or "")
    files = {a: os.path.normpath(os.path.join(src_meshdir, a.get("file"))) for a in root.iter("mesh")
             if a.get("file") and not os.path.isabs(a.get("file"))}
    if files:
        common = os.path.commonpath(list(files.values()))
        comp.set("meshdir", os.path.relpath(common, OUT) + "/")
        for a, f in files.items():
            a.set("file", os.path.relpath(f, common))
    root.set("model", f"g1_29dof_brainco_ultra_{effort_set}")

    kf = root.find("keyframe")
    if kf is not None:
        root.remove(kf)

    # fingers: drop the 12 position actuators, hold the pose with passive springs
    act = root.find("actuator")
    finger_joints = {}
    for a in list(act):
        if a.tag != "motor":
            lo, hi = (float(v) for v in a.get("ctrlrange").split())
            finger_joints[a.get("joint")] = lo + finger_curl * (hi - lo)
            act.remove(a)
    for j in root.iter("joint"):
        if j.get("name") in finger_joints:
            j.set("stiffness", str(FINGER_SPRING))
            j.set("springref", f"{finger_joints[j.get('name')]:.4f}")

    # body motors: effort set
    spec = json.load(open(SPEC))
    efforts = spec["effort_limit_ultra"] if effort_set == "ultra" else spec["effort_limit"]
    names = spec["joint_names"]
    motors = [a for a in act if a.tag == "motor"]
    assert [a.get("joint") for a in motors] == names, "motor order != contract order"
    for a, e in zip(motors, efforts):
        a.set("ctrlrange", f"{-e:g} {e:g}")
    joints = {j.get("name"): j for j in root.iter("joint")}
    for n, e in zip(names, efforts):     # the robot file also caps the force at the joint (actuatorfrcrange)
        joints[n].set("actuatorfrcrange", f"{-e:g} {e:g}")

    # ULTRA's simulator settings: solver options and the ground plane (with its material)
    u = ET.parse(ULTRA_G1).getroot()
    root.remove(root.find("option"))
    root.insert(list(root).index(comp) + 1, u.find("option"))
    asset = root.find("asset")
    for tag in ("texture", "material"):
        for e in (x for a in u.findall("asset") for x in a.findall(tag)):
            if e.get("name") in ("texplane", "MatPlane"):
                asset.append(e)
    world = root.find("worldbody")
    world.insert(0, next(g for w in u.findall("worldbody") for g in w.findall("geom") if g.get("name") == "ground"))

    # ULTRA marker bodies (massless, no geoms): kinematic frames only, so the observation matches training
    bodies = {b.get("name"): b for b in root.iter("body")}
    for name, (parent, pos, quat) in ultra_marker_frames().items():
        ET.SubElement(bodies[parent], "body", name=name,
                      pos=" ".join(f"{v:.6g}" for v in pos), quat=" ".join(f"{v:.6g}" for v in quat))

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"g1_brainco_ultra_{effort_set}.xml")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(out)
    assert m.nu == 29, m.nu
    print(f"{out}: nq={m.nq} nu={m.nu} nbody={m.nbody} mass={m.body_subtreemass[1]:.3f} kg, "
          f"ankle effort {m.actuator_ctrlrange[4][1]:g} Nm, fingers at {finger_curl:.2f} curl")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finger-curl", type=float, default=0.0, help="0 = open hand, 1 = fully closed")
    args = ap.parse_args()
    for effort_set in ("model", "ultra"):
        build(effort_set, args.finger_curl)
