# ULTRA's released policy on our G1 + BrainCo (MuJoCo)

Tests the tracking teacher released with [ULTRA](https://github.com/Sirui-Xu/ULTRA)
(`ultra/weights/teacher_ultra_inference.pth`) on our G1 29-DoF rev 1.0 with BrainCo Revo2 hands, in MuJoCo,
against ULTRA's own G1 as the baseline. No training; this only runs their weights.

Paths in the table are relative to this folder (`ultra_eval/`); `../../datasets/` is the hospital-sim root.

| Path | What |
|---|---|
| `../third_party/ULTRA/` | ULTRA, full git clone (remote `upstream`, commit `ec972c7`), **unmodified** |
| `../../datasets/ultra/OMOMO_retarget_aug_v4/` | ULTRA's released retargeted + augmented G1 motions (4,066 clips, 3.3 GB, git-ignored) |
| `build_ultra_robot.py` | generates `assets/g1_brainco_ultra_{model,ultra}.xml`, our robot in the form the policy observes |
| `eval_teacher.py` | batch evaluator (track / stand / push), reuses ULTRA's sim2sim code |
| `run_upstream.py` | runs any ULTRA sim2sim script unchanged on a CPU-only torch |
| `run.sh` | runs the above in conda env `ultra-sim2sim`, pinned to the good CPUs |
| `results/*.json` | per-clip results; `media/*.mp4` example videos |

## What the released policy is

- **Teacher only.** ULTRA publishes one network: the motion-tracking teacher (4052-D observation → 29 joint
  targets, 60 Hz, MLP 1024-1024-512). The deployable students (perception + sparse goals) are not released.
- **It needs a reference motion.** It reproduces a recorded human-object interaction clip (ULTRA's `[T, 630]`
  format). Without a clip it has nothing to track; "standing" = tracking a held standing frame.
- **It observes privileged state**: world body positions/velocities of 39 bodies and the reference. Fine in
  simulation; a real robot would need full state estimation (mocap) to run it. Real deployment needs a student.
- **Licence**: the weights are for non-commercial research and education (trained on AMASS and BONES-SEED
  motions, see `third_party/ULTRA/docs/teacher-model-card.md`).
- In ULTRA's MuJoCo sim2sim the teacher's object observation is masked (as in 70 % of its training), so it
  tracks the body motion only; box carrying is not expected to work there, on either robot.

## How our robot is run

`build_ultra_robot.py` keeps our robot's dynamics (33.8 kg; BrainCo hands 0.39 kg each vs 0.17 kg rubber hands;
our inertias, joint limits, contact geometry) and adds only what the policy needs to see the same observation:

- massless marker bodies with ULTRA's names at ULTRA's frames (rubber_hand, head, IMUs, camera, lidar);
  world positions of all 39 observed bodies match ULTRA's G1 exactly at the same joint angles;
- fingers unactuated, held open by passive springs (`--finger-curl` to pre-curl), so there are 29 motors;
  the policy's hand contact flag is set by contact anywhere on the BrainCo palm or fingers;
- ULTRA's ground plane and solver settings (PGS, 50 iterations; floor friction 2.0);
- motor and joint force limits from the chosen effort set: `model` = 35 N·m ankles and waist roll/pitch
  (this project's choice), `ultra` = 50 N·m (what ULTRA trained with). The PD torque is clipped at ULTRA's
  0.8 × 50 N·m, and never above the robot's limit.

Everything else is ULTRA's code and constants: observation builder, motion and object loading, reset, network
and checkpoint, gains, 1020 Hz PD / 60 Hz policy, action scale.

## Results (2026-10-09)

Fall = pelvis below 0.45 m, or mean body-position error above 0.5 m. MPBPE = mean distance between the 39
observed bodies and the reference, survivors only. Brackets: 95 % Wilson interval.

**Tracking, 100 random clips** (same clips for every robot; each clip is tracked, then its last frame is held
for as long again):

| Robot | Survived | MPBPE in motion | MPBPE holding | Joint error | Object error |
|---|---|---|---|---|---|
| ULTRA's G1, rubber hands, 50 N·m | 91/100 [84–95 %] | 5.8 cm | 4.2 cm | 0.066 rad | 45 cm |
| Our G1 + BrainCo, 50 N·m | 88/100 [80–93 %] | 5.7 cm | 4.0 cm | 0.066 rad | 52 cm |
| Our G1 + BrainCo, 35 N·m | 90/100 [83–94 %] | 5.6 cm | 3.9 cm | 0.066 rad | 51 cm |

**Standing still for 20 s next to the object, 50 clips**, with three 0.2 s horizontal pelvis pushes in random
directions (100 / 200 / 300 N ≈ 0.6 / 1.2 / 1.8 m/s velocity change for 33.8 kg):

| Robot | No push | 100 N | 200 N | 300 N |
|---|---|---|---|---|
| ULTRA's G1, rubber hands, 50 N·m | 50/50 | 50/50 | 44/50 [76–94 %] | 5/50 |
| Our G1 + BrainCo, 50 N·m | 50/50 | 50/50 | 46/50 [81–97 %] | 10/50 |
| Our G1 + BrainCo, 35 N·m | 50/50 | 50/50 | 44/50 [76–94 %] | 6/50 |

Standing body error is 3.5 cm without pushes on all three; torque saturation stays below 0.3 % of steps.

**Reading:**

- The released teacher runs on our G1 + BrainCo **as well as on ULTRA's own G1**: survival and tracking error
  are within each other's confidence intervals. The heavier BrainCo hands make no measurable difference.
- **35 N·m vs 50 N·m makes no measurable difference** for these motions or for standing and pushes up to 200 N.
  Peaks above 35 N·m happen (ankle pitch, waist pitch) but are brief.
- It stands indefinitely and recovers 100 N pushes every time and 200 N pushes ~90 % of the time. 300 N pushes
  (≈ 1.8 m/s) knock it over most of the time on every robot: it was trained to track, not to step to recover.
- The ~10 % tracking failures are the same clips on every robot (sim2sim gap, not our robot).
- Box carrying fails on every robot in this mode (object observation masked, see above).

Example videos (our robot, 35 N·m): `media/ours_35Nm_track_largebox.mp4` (crouches to the box, tips it, stands
back up: body tracked, box not carried) and `media/ours_35Nm_stand_push200N.mp4` (stands, then falls after a
200 N push: one of the ~10 % failures at that level).

## What it means for this project

- Use ULTRA's teacher as a **reference / teacher** in simulation, e.g. to check retargeted motions on our robot,
  or as the expert for distilling a student. It is not something to put on the real robot directly.
- For standing during surgery and teleop, the plan in `../README.md` (Stage A stand, Stage B teleop, with a
  deployable student and pushes in training) still applies; this test shows our robot model and the 35 N·m
  limit are not a problem for ULTRA-style control.

## Reproduce

```bash
# one-time: conda env (pin to the good CPUs; unpinned installs corrupted files on this machine)
taskset -c 0-7,10-31 conda create -y -n ultra-sim2sim python=3.10
taskset -c 0-7,10-31 ~/miniconda3/envs/ultra-sim2sim/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
taskset -c 0-7,10-31 ~/miniconda3/envs/ultra-sim2sim/bin/pip install "rl-games==1.1.4" "mujoco>=3.0,<3.3" trimesh scipy tqdm "imageio[ffmpeg]" pyyaml gym tensorboardX gdown "numpy<2"
# data (2.8 GB): ../datasets/ultra/OMOMO_retarget_aug_v4/ (hospital-sim root, next to this repo)
(cd ../datasets/ultra && gdown 1g6OzxXGJczZ4klVqKCpTzy-orKkllCr4 && unzip OMOMO_retarget_aug_v4*.zip)

# the rest from the g1-whole-body-rl root
ultra_eval/run.sh build_ultra_robot.py
ultra_eval/run.sh eval_teacher.py --robot ultra --clips 100 --workers 20 --out ultra_eval/results/track_ultra_ultra.json
ultra_eval/run.sh eval_teacher.py --robot ours --effort model --clips 100 --workers 20 --out ultra_eval/results/track_ours_model.json
ultra_eval/run.sh eval_teacher.py --robot ours --effort model --mode stand --duration 20 --push 3 --push-force 200 \
    --clips 50 --seed 1 --workers 20 --out ultra_eval/results/stand_ours_model_push200.json
# a video of one clip: add --clip <file.pt> --workers 1 --video ultra_eval/media/x.mp4
# ULTRA's own script, unchanged:
cd third_party/ULTRA && PYTHONPATH=ultra ../../ultra_eval/run.sh run_upstream.py ultra/sim2sim_teacher.py \
    --ckpt ultra/weights/teacher_ultra_inference.pth --motion_path <clip.pt> --record_video
```
