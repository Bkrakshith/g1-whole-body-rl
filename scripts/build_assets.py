#!/usr/bin/env python
"""Generate assets/g1_mjx.xml (training, MJX) and assets/g1_eval.xml
(evaluation, plain CPU MuJoCo) from the robot model named in
configs/robot.yaml. Mechanical and repeatable: NEVER hand-edit the outputs;
re-run this after g1-brainco-description/ changes or after editing configs/actuators.yaml,
then `pytest tests/test_model.py`.

Source selection: the first entry of configs/robot.yaml `sources` whose train
file exists (g1-brainco-description/ canonical first, the unitree_rl_gym rev 1.0 model as a
clearly marked FALLBACK). The choice is written to assets/PROVENANCE.md and
assets/model_info.json, and propagated into every exported policy.json.

Both outputs:
  * the 29 body joints in docs/interfaces.md order; ONE motor per body joint,
    actuator order = that order, ctrlrange = +-0.8 x effort (ULTRA
    humanoid_g1.py torque_limits), armature = configs/actuators.yaml (ULTRA);
  * the SAME foot-floor contact: 4 sole spheres per ankle_roll_link (the
    source's own sole spheres if it has them, else the Unitree rev 1.0 ones)
    -- skill failure M14: train/eval sole geometry must match;
  * a floor, a `home` keyframe = actuators.yaml default_pose with the pelvis
    height set so the soles touch the floor; timestep 0.002 (contract sim_dt);
  * sensors, cameras and source keyframes removed.
g1_mjx.xml additionally drops every other geom (visual meshes and mesh
collisions: MJX cost scales with contact pairs) and uses the MJX-friendly
solver settings validated on the H1-2/X1 ports (Newton, 2 iters, pyramidal).
g1_eval.xml keeps the source's mesh collisions (full fidelity, catches
policies that exploit primitive-geometry artefacts); hand joints of the full
BrainCo model (if any) get motors too and are PD-held open by the evaluator.
"""

import argparse
import json
import os
import re
import sys

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.dirname(RL)          # hospital-sim master checkout (sibling repos live here)
sys.path.insert(0, os.path.join(RL, "src"))

from g1_wbc.layout import JOINT_NAMES, torque_limit  # noqa: E402

# Unitree G1 rev 1.0 sole spheres (unitree_rl_gym g1_29dof_rev_1_0.xml,
# ankle_roll_link frame): used only if the source has no sole spheres.
UNITREE_SOLE = {"radius": 0.005,
                "pos": [[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03],
                        [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]]}
FLOOR_FRICTION = [1.0, 0.02, 0.01]   # ULTRA cfg plane static/dynamic friction 1.0


def _resolve(p):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(RL, p))


def _show(p):
    """Path as written into generated files: relative to the master checkout
    when inside it (the checkout moves between machines), else with ~."""
    if os.path.commonpath([p, ROOT]) == ROOT:
        return os.path.relpath(p, ROOT)
    home = os.path.expanduser("~")
    return "~" + p[len(home):] if p.startswith(home + os.sep) else p


def pick_source():
    with open(os.path.join(RL, "configs", "robot.yaml")) as f:
        cfg = yaml.safe_load(f)
    for s in cfg["sources"]:
        tr, ev, eh = _resolve(s["train"]), _resolve(s["eval"]), _resolve(s["eval_hands"])
        if all(os.path.exists(p) for p in (tr, ev, eh)):
            return s["name"], tr, ev, eh, cfg
    raise SystemExit("no robot source found; see configs/robot.yaml")


def _load(path):
    spec = mujoco.MjSpec.from_file(path)
    base = os.path.dirname(path)
    if spec.meshdir and not os.path.isabs(spec.meshdir):
        spec.meshdir = os.path.normpath(os.path.join(base, spec.meshdir))
    elif not spec.meshdir:
        spec.meshdir = base
    if spec.texturedir and not os.path.isabs(spec.texturedir):
        spec.texturedir = os.path.normpath(os.path.join(base, spec.texturedir))
    return spec


def _relative_asset_dirs(xml, out_dir):
    """meshdir/texturedir relative to the output file (the checkout must stay
    relocatable). MjSpec resolves relative dirs against the cwd when it
    compiles, so the spec keeps absolute dirs and only the text is rewritten."""
    def _rel(m):
        d = m.group(2)
        if os.path.isabs(d):
            d = os.path.relpath(d, out_dir) + "/"
        return f'{m.group(1)}="{d}"'
    return re.sub(r'(meshdir|texturedir)="([^"]*)"', _rel, xml)


def _strip_scene(spec):
    # equalities (BrainCo finger mimic couplings) and contact excludes are
    # physics and stay; sensors/keys/cameras/actuators/lights are replaced.
    for coll in (spec.sensors, spec.keys, spec.cameras, spec.actuators, spec.lights):
        for el in list(coll):
            spec.delete(el)
    for g in list(spec.worldbody.geoms):          # old floor etc.
        spec.delete(g)


def _sole_spheres(spec):
    """(radius, [pos...]) of the source's sole spheres on the LEFT foot (the
    right is checked to mirror it), else the Unitree rev 1.0 defaults."""
    out = {}
    for side in ("left", "right"):
        b = spec.body(f"{side}_ankle_roll_link")
        sph = [g for g in b.geoms if g.type == mujoco.mjtGeom.mjGEOM_SPHERE
               and g.contype != 0]
        out[side] = sph
    if len(out["left"]) >= 3 and len(out["left"]) == len(out["right"]):
        r = float(out["left"][0].size[0])
        return {side: {"radius": r, "pos": [list(map(float, g.pos)) for g in out[side]]}
                for side in out}, "source MJCF sole spheres"
    return {side: dict(UNITREE_SOLE) for side in ("left", "right")}, \
        "Unitree rev 1.0 sole spheres (source had none)"


def _set_soles(spec, soles):
    """Replace every collision geom on each ankle_roll_link by the sole spheres
    and drop the ankle_pitch_link collision (it can reach the floor under a
    large ankle roll; the soles must be the ONLY foot-floor contact, M14)."""
    for side in ("left", "right"):
        for g in list(spec.body(f"{side}_ankle_pitch_link").geoms):
            if g.contype != 0 or g.conaffinity != 0:
                spec.delete(g)
        b = spec.body(f"{side}_ankle_roll_link")
        for g in list(b.geoms):
            if g.contype != 0 or g.conaffinity != 0:
                spec.delete(g)
        for k, p in enumerate(soles[side]["pos"]):
            b.add_geom(name=f"{side}_sole_{k}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[soles[side]["radius"], 0, 0], pos=p, contype=2,
                       conaffinity=1, condim=3, friction=FLOOR_FRICTION, group=3,
                       rgba=[0.9, 0.3, 0.3, 1])
        pts = np.asarray(soles[side]["pos"])
        r = soles[side]["radius"]
        c = pts.mean(axis=0)
        c[2] -= r
        b.add_site(name=f"{side}_foot", pos=c.tolist(), size=[0.01, 0, 0])
        # NON-colliding reference box over the sole: support polygon for the
        # skill's posture_ik / reach analysis and the env's foot frame.
        half = [(pts[:, 0].max() - pts[:, 0].min()) / 2 + r,
                (pts[:, 1].max() - pts[:, 1].min()) / 2 + r, 0.005]
        b.add_geom(name=f"{side}_foot", type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
                   pos=[c[0], c[1], c[2] + 0.005], contype=0, conaffinity=0,
                   group=4, rgba=[0.7, 0.3, 0.3, 0.2])


def _joint_order_check(spec):
    m = spec.compile()
    hinge = [m.joint(j).name for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    body = [n for n in hinge if n in JOINT_NAMES]
    if body != list(JOINT_NAMES):
        missing = sorted(set(JOINT_NAMES) - set(hinge))
        raise SystemExit(f"source joint order/names differ from the contract: "
                         f"missing={missing} order={body}")
    extra = [n for n in hinge if n not in JOINT_NAMES]
    return extra


def _actuators(spec, act, extra_joints):
    eff = np.asarray(torque_limit(act))
    for i, name in enumerate(JOINT_NAMES):
        a = spec.add_actuator(name=name, target=name,
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = [-eff[i], eff[i]]
        j = spec.joint(name)
        j.armature = float(act["armature"][i])
        j.frictionloss = 0.0
        j.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
    for name in extra_joints:          # BrainCo finger joints (eval model only)
        a = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = [-2.0, 2.0]
        spec.joint(name).armature = max(spec.joint(name).armature, 1e-4)


def _scene(spec, mjx):
    spec.option.timestep = 0.002
    if mjx:
        spec.option.iterations = 2
        spec.option.ls_iterations = 4
        spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
        spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], contype=1, conaffinity=2 | (0 if mjx else 1),
                            condim=3, friction=FLOOR_FRICTION, rgba=[0.3, 0.35, 0.4, 1])
    spec.worldbody.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)


def _strip_for_mjx(spec, keep_prefix=("left_sole_", "right_sole_", "left_foot", "right_foot", "floor")):
    for b in spec.bodies:
        for g in list(b.geoms):
            if not (g.name or "").startswith(keep_prefix):
                spec.delete(g)
    for el in list(spec.excludes) + list(spec.pairs):
        spec.delete(el)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    for tex in list(spec.textures):
        spec.delete(tex)
    for mat in list(spec.materials):
        spec.delete(mat)


def _home(spec, default_pose):
    """Add the `home` keyframe; return (model, data at home)."""
    m = spec.compile()
    d = mujoco.MjData(m)
    q = np.zeros(m.nq)
    q[3] = 1.0
    for i, name in enumerate(JOINT_NAMES):
        q[m.jnt_qposadr[m.joint(name).id]] = default_pose[i]
    q[2] = 1.0
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    soles = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(("left_sole_", "right_sole_"))]
    lowest = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in soles)
    q[2] = 1.0 - lowest
    spec.add_key(name="home", qpos=q.tolist())
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def _model_info(m, d):
    lf, rf = d.site_xpos[m.site("left_foot").id], d.site_xpos[m.site("right_foot").id]
    hands = {}
    for side in ("left", "right"):
        for cand in (f"{side}_wrist_yaw_link", f"{side}_rubber_hand"):
            try:
                hands[side] = d.xpos[m.body(cand).id].tolist()
                break
            except KeyError:
                continue
    return {
        "pelvis_height_home_m": float(d.qpos[2]),
        "total_mass_kg": float(m.body_subtreemass[1]),
        "feet_lateral_home_m": float(abs(lf[1] - rf[1])),
        "feet_xy_home_m": [lf[:2].tolist(), rf[:2].tolist()],
        "com_home_m": d.subtree_com[1].tolist(),
        "hand_pos_home_m": hands,
        "torso_height_home_m": float(d.xpos[m.body("torso_link").id][2]),
    }


def build(verbose=True):
    name, src_train, src_eval, src_hands, rcfg = pick_source()
    with open(os.path.join(RL, "configs", "actuators.yaml")) as f:
        act = yaml.safe_load(f)
    outs = {}
    for kind, src, mjx in (("train", src_train, True), ("eval", src_eval, False),
                           ("eval_hands", src_hands, False)):
        spec = _load(src)
        spec.modelname = f"g1_29dof_{'mjx' if mjx else 'eval'}"
        _strip_scene(spec)
        soles, sole_src = _sole_spheres(spec)
        _set_soles(spec, soles)
        extra = _joint_order_check(spec)
        if mjx and extra:
            raise SystemExit(f"training model must have exactly the 29 body joints; "
                             f"extra hinge joints {extra} (use the fixedhands model)")
        _actuators(spec, act, extra)
        if mjx:
            _strip_for_mjx(spec)
        _scene(spec, mjx)
        m, d = _home(spec, act["default_pose"])
        out = _resolve(rcfg[f"out_{kind}"])
        xml = _relative_asset_dirs(spec.to_xml(), os.path.dirname(out))
        with open(out, "w") as f:
            f.write(f"<!-- GENERATED by scripts/build_assets.py from {_show(src)} "
                    f"(source '{name}'). Do not edit. -->\n" + xml)
        mujoco.MjModel.from_xml_path(out)  # round-trip check
        outs[kind] = dict(path=os.path.relpath(out, RL), src=_show(src), nq=m.nq, nv=m.nv,
                          nu=m.nu, ngeom=m.ngeom, extra_joints=extra, sole=sole_src,
                          info=_model_info(m, d))
    info = {"source_name": name, "fallback": name.startswith("FALLBACK"), **outs}
    with open(os.path.join(RL, "assets", "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    _provenance(info, act)
    if verbose:
        print(json.dumps(info, indent=2))
    return info


def _provenance(info, act):
    t, e = info["train"], info["eval"]
    lines = [
        "# assets provenance (GENERATED by scripts/build_assets.py)", "",
        f"* robot source: **{info['source_name']}**"
        + ("  <-- FALLBACK: g1-brainco-description canonical model not present at build time; "
           "rebuild when it lands" if info["fallback"] else ""),
        f"* training model `{t['path']}` from `{t['src']}`: "
        f"nq {t['nq']}, nu {t['nu']}, ngeom {t['ngeom']}",
        f"* eval model `{e['path']}` from `{e['src']}`: "
        f"nq {e['nq']}, nu {e['nu']}, ngeom {e['ngeom']}, extra (hand) joints "
        f"{len(e['extra_joints'])}",
        f"* eval-with-hands model `{info['eval_hands']['path']}` from "
        f"`{info['eval_hands']['src']}`: nq {info['eval_hands']['nq']}, nu {info['eval_hands']['nu']}, "
        f"finger joints {len(info['eval_hands']['extra_joints'])} (PD-held open; joints addressed BY NAME)",
        f"* foot contact (all models, identical, skill M14): {t['sole']}",
        f"* home: pelvis {t['info']['pelvis_height_home_m']:.4f} m, mass "
        f"{t['info']['total_mass_kg']:.2f} kg, feet lateral "
        f"{t['info']['feet_lateral_home_m']:.4f} m (eval model mass "
        f"{e['info']['total_mass_kg']:.2f} kg)",
        "* gains/armature/effort: configs/actuators.yaml (ULTRA humanoid_g1.py); "
        "ctrlrange = 0.8 x effort", "",
        "Regenerate: `./run.sh scripts/build_assets.py` then `./run.sh -m pytest tests/test_model.py`.",
    ]
    with open(os.path.join(RL, "assets", "PROVENANCE.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
