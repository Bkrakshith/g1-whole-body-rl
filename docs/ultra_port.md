# ULTRA port: parameter map and deviations

These notes come from the README design notes. ULTRA is github.com/Sirui-Xu/ULTRA (arXiv 2603.03279),
checked out as the submodule `third_party/ULTRA`. The table uses three abbreviations: "hg1" =
`ultra/env/tasks/humanoid_g1.py`, "ug1" = `ultra/env/tasks/ultra_g1.py`, "rlg" = ULTRA's rl_games configs.
Paths in the right column are relative to this repository.

## 1. Parameter map

The table lists every ULTRA parameter, its value, and where it lives in this repository.

| ULTRA source (file:key) | value | here |
|---|---|---|
| hg1 `stiffness` | 40.18/99.10/28.50/14.25/16.78 per DoF | `configs/actuators.yaml` kp |
| hg1 `damping` | 2.558/6.309/1.814/0.907/1.068 | `configs/actuators.yaml` kd |
| hg1 `armature` | 0.010178 ... 0.00425 | `configs/actuators.yaml` armature, written into both MJCFs |
| hg1 `effort`, `torque_limits = effort*0.8` | 88/139/88/139/50/50, waist 88/50/50, arms 25.../5/5; x0.8 | `actuators.yaml` effort_ultra, torque_limit_fraction (canonical 35 N m used, see section 2 item 8 and the README) |
| hg1 `action_scale = effort/stiffness` (computed, unused by ULTRA) | 2.19/1.40/1.75 | x0.25 -> legs/waist action_scale |
| g1_teacher.yaml control.action_scale (docs/deploy.md "action * 3.0") | 3 | arms action_scale |
| ug1 `init_dof` | legs -0.1/0/0/0.3/-0.2/0, waist 0, arms 0/0/0/0.5/0/0/0 | default_pose, `home` keyframe |
| g1_teacher.yaml env.controlFrequencyInv 17 @ 1 ms | 58.8 Hz | NOT ported: contract 50 Hz / 500 Hz |
| domain_rand friction_range | [0.5, 2.0] | `configs/domain_rand.yaml` ground (floor + soles) |
| added_mass_range / added_com_range (torso_link) | +-3 kg / +-0.05 m | `domain_rand.yaml` body |
| push_robots, push_interval_s, max_push_vel_xy | every 4 s, set base v_xy U(+-1.0) | `domain_rand.yaml` push (velocity kick) |
| motor_strength_range (kp, kd per DoF) | [0.8, 1.2] | `domain_rand.yaml` actuation |
| action_delay (randint(2)) | 0-1 control step | `domain_rand.yaml` actuation |
| gravity_range / gravity_rand_interval_s | +-0.1 m/s^2 / 4 s | `domain_rand.yaml` gravity |
| noise.noise_scales (uniform, noise_level 1) | dof_pos .01, dof_vel .1, ang_vel .05, imu .05 | `domain_rand.yaml` obs_noise (uniform, student only) |
| smooth_rewards.scales | action_rate -0.1, torques -0.001, dof_vel -4e-4, dof_pos_limits -5, dof_torque_limits -1, energy -1e-4, delta_dof_vel_all -2e-5, delta_ang_vel_all -5e-4, feet_orientation -0.35, foot_slip -0.1, feet/knee_distance -0.1, stand_on_feet -1, termination -50 | Stage C: raw inside exp(S/15); Stage A/B: /15 additive (`configs/rewards.yaml`) |
| smooth_rewards min/max_dist, soft_torque_limit | 0.25/0.65, 0.95 | `rewards.yaml` (min 0.20, see item 5 below) |
| rewardWeights p/r/pv/eg1 | 10/5/0.1/0 | `configs/stage_c_tracking.yaml` |
| env.scaling, terminationHeight, stateInit Hybrid, hybridInitProb, keyBodies | 0.8, 0.15, Hybrid, 0.1, 14 bodies | `stage_c_tracking.yaml` |
| rlg g1_teacher: mlp [1024,1024,512] relu, gamma .99, tau .95, e_clip .2, grad_norm 1, normalize_input | | `configs/train.yaml` networks/ppo (gamma 0.98 A/B, 0.988 C) |
| rlg g1_teacher: lr 2e-5, horizon 32, minibatch 16384, mini_epochs 6, entropy 0 | | `stage_c_tracking.yaml` ppo |
| sigma_init -2.9 (fixed std) | 0.055 x 3 = 0.165 rad | `train.yaml` networks.init_joint_noise_rad (initial per-joint std) |
| g1_student_vae.yaml history_len 10; rlg vae_dim 64, lr 2e-4 -> 5e-5, mini_epochs 2, expert_loss_coef 1 | | `train.yaml` student / distill |
| ultra_agent_distill_vae.py beta, kld 0.001->0.2, prior-action 0->0.6, smooth 1e-4->1e-3 | | `train.yaml` distill |
| ULTRA README data links | Google Drive | `stage_c_tracking.yaml` header |

## 2. Deviations from ULTRA

Nine items differ from ULTRA. Each item gives the reason.

1. The policy runs at 50 Hz and the PD loop at 500 Hz, instead of 58.8 Hz / 1 kHz. The contract fixes
   control_dt 0.02 and sim_dt 0.002.
2. The action is default + a*scale with per-joint scales (legs and waist 0.25*effort/kp, arms 3.0), instead
   of `3*a` absolute for all joints. The reason: brax PPO learns its std (about 0.7 at init), so 3 rad leg
   scales would inject about 2 rad of noise at step 0. The code sets the initial std per joint to match
   ULTRA's 0.165 rad joint noise, and initialises the policy's last layer at x0.01.
3. The Stage A/B reward is additive: the unitree_rl_gym G1 locomotion terms, the skill's torso-stability
   terms, and the ULTRA smooth pack at scale/15. The factor 1/15 is the first-order expansion of ULTRA's
   multiplicative form. ULTRA has no velocity-command teleop mode to port. Stage C keeps ULTRA's
   multiplicative form exactly.
4. The student observation uses projected gravity instead of roll/pitch Euler angles. Projected gravity has
   no singularity and the same noise amplitude.
5. The minimum feet and knee distance is 0.20 m. ULTRA's 0.25 m is above the G1's own 0.237 m stance.
6. The code draws the action delay and the push phase per environment and per episode. ULTRA draws one
   delay and one push tick for all environments.
7. Stage C drops the objects, the interaction graph, the hand-contact terms and the contact-force terms,
   because this MJX scene has no objects or contact forces. It drops the `mid360_link` key body and maps
   `rubber_hand` to `wrist_yaw_link`. The port reduces ULTRA's multimodal transformer student (point clouds,
   modality masks) to a proprioceptive-history CVAE.
8. Training clips the torque at 0.8 x effort; deployment uses the full effort. The controller `g1_controller`
   loads `torque_limit`, but the motor driver and the simulator clip at full effort. So deployment has 25 %
   more authority than training. This is the conservative direction, and ULTRA did the same.
9. Brax PPO has no critic_coef or bounds_loss. Stage A/B use the skill-validated brax PPO settings (lr 3e-4,
   entropy 5e-3, 4 epochs) rather than rl_games' lr 2e-5.
