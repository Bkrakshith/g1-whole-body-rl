#!/usr/bin/env python
"""Battery on the full-fidelity eval model (plain CPU MuJoCo), Wilson CIs.

  ./run.sh eval/run_battery.py --export exports/g1_teleop_v1 [--episodes 50] [--hands]
  ./run.sh eval/run_battery.py --teacher outputs/checkpoints/teleop_v1/ckpt_N
  ./run.sh eval/run_battery.py --student outputs/checkpoints/teleop_student_v1/ckpt_best
--export uses ONLY policy.onnx + policy.json (the deployment path). --hands runs
on the full BrainCo model (finger joints held open). --gate G exits 1 if any
scenario's survival < G (ONNX battery gate before hand-off, skill 9).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--export")
    g.add_argument("--teacher")
    g.add_argument("--student")
    ap.add_argument("--which", choices=["stand", "teleop"], default=None)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--hands", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of scenario names")
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from g1_wbc import battery, sim2sim
    from g1_wbc.config import ASSETS_DIR, RL_ROOT, load_config
    cfg = load_config()
    if a.export:
        pol = sim2sim.OnnxPolicy(a.export)
        sidecar = pol.sc
        which = a.which or sidecar["training"]["curriculum"]
    else:
        sys.path.insert(0, os.path.join(RL_ROOT, "export"))
        from export_policy import sidecar as make_sidecar
        pol = (sim2sim.TeacherPolicy(a.teacher, cfg) if a.teacher
               else sim2sim.StudentCkptPolicy(a.student, cfg))
        sidecar = make_sidecar(cfg, "ckpt", {}, "none.onnx", int(cfg.train.student.history_len))
        which = a.which or "teleop"
    xml = os.path.join(ASSETS_DIR, "g1_eval_hands.xml" if a.hands else "g1_eval.xml")
    sim = sim2sim.Sim2Sim(sidecar, xml, cfg)
    scen = battery.scenarios(cfg, which)
    if a.only:
        scen = [s for s in scen if s.name in a.only.split(",")]
    report, worst = {}, 1.0
    for sc in scen:
        rs = []
        for ep in range(a.episodes):
            # a fresh random schedule per episode (kick directions, poses)
            sc_ep = [s for s in battery.scenarios(cfg, which, seed=ep) if s.name == sc.name][0]
            rs.append(sim.run(sc_ep, pol, seed=ep))
        k = sum(r.survived for r in rs)
        lo, hi = sim2sim.wilson(k, len(rs))
        report[sc.name] = {"survival": k / len(rs), "ci95": [lo, hi], "n": len(rs),
                           "mean_t": sum(r.t_survived for r in rs) / len(rs),
                           "vel_err": sum(r.vel_err for r in rs) / len(rs),
                           "upper_rms_rad": sum(r.upper_rms for r in rs) / len(rs),
                           "tilt_max_deg": max(r.tilt_deg_max for r in rs),
                           "torque_sat": sum(r.torque_sat_frac for r in rs) / len(rs)}
        worst = min(worst, k / len(rs))
        r = report[sc.name]
        print(f"{sc.name:18s} survival {r['survival']:.2f} [{lo:.2f},{hi:.2f}] n={len(rs)} "
              f"vel_err {r['vel_err']:.3f} upper_rms {r['upper_rms_rad']:.3f} "
              f"tilt_max {r['tilt_max_deg']:.1f} deg sat {r['torque_sat']:.3f}", flush=True)
    out = a.out or os.path.join(RL_ROOT, "outputs", "eval",
                                (os.path.basename(a.export or a.teacher or a.student).rstrip("/")
                                 + ("_hands" if a.hands else "") + ".json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"model": xml, "scenarios": report}, open(out, "w"), indent=2)
    print(f"-> {out}")
    if a.gate is not None and worst < a.gate:
        print(f"GATE FAILED: worst survival {worst:.2f} < {a.gate}")
        sys.exit(1)


if __name__ == "__main__":
    main()
