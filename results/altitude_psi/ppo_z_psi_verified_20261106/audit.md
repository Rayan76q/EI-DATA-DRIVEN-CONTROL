# PPO altitude and yaw audit

The selected PPO completes the raw-RNN climb, full turn, and soft landing mission.
Training reached 1,501,184 total environment steps.
The selected successful policy is from step 266,240; later updates do not replace a better checkpoint.

## Changes

- Fixed the failure reward sign: failures always receive a negative terminal reward.
- Used positive altitude/yaw tracking rewards with smaller yaw-effort and command-change penalties.
- Kept mission angles in radians: 0 m until 2 s, 2 m until 8 s, a full turn from 5 to 8 s, then landing at -0.3 m while holding 2π. The deadline is 20 s.
- Success now requires at least 0.5 s within 0.2 m of the positive altitude target, final yaw within 10 degrees of the unwrapped goal, touchdown speed at most 0.5 m/s, and yaw speed at most 0.5 rad/s. A skipped climb cannot qualify.
- Scaled PPO observations and actions to balance thrust and yaw learning. Physical commands are `thrust = 0.4 + a[0]`, `yaw = 0.1*a[1]`, where PPO actions are in [-1, 1]. Those commands still enter the raw environment as `[thrust, 0, 0, yaw]`.
- Saved successful checkpoints ahead of failed evaluations. Saved action scaling and RNN/normalization identity with checkpoints; incompatible resumes/inference are rejected.
- Updated notebook inference and export to apply the saved action transform once. Editable heuristic formulas were preserved.

The RNN weights and raw dynamics function were not changed during this audit. Normalization uses the current values supplied in `droneEnvRotation.PARAMS`, not the older values in previous runs. No extra gravity, ground support, rotation matrix, hidden-state warmup, heuristic controller, or other RL algorithm is added to PPO evaluation.

## Default mission result

| Measure | Result |
|---|---:|
| Outcome | landed, has_turned |
| Duration | 11.85 s |
| Peak altitude | 1.901 m |
| Time within altitude tolerance | 2.85 s |
| Final yaw | 358.63 degrees |
| Touchdown vertical speed | -0.071 m/s |
| Final yaw speed | 0.0086 rad/s |
| Episode return | 195.97 |

The previous saved policy crashed at 5.55 s and reached only 0.032 m under the current RNN and mission. Its original training dynamics may differ, so this is an operational comparison, not an isolated reward ablation.

## Additional mission checks

| Altitude target (m) | Turn (degrees) | Success | Duration (s) | Touchdown vz (m/s) |
|---:|---:|:---:|---:|---:|
| 1 | 360 | yes | 11.15 | -0.071 |
| 2 | 360 | yes | 11.85 | -0.071 |
| 3 | 360 | yes | 12.30 | -0.070 |
| 2 | -360 | yes | 12.10 | -0.070 |
| 2 | 180 | yes | 11.90 | -0.070 |
| 2 | -180 | yes | 12.00 | -0.070 |
| 1 | 0 | yes | 11.30 | -0.069 |
| 3 | 0 | yes | 12.35 | -0.071 |

These are eight deterministic mission variants using the same learned RNN. Repeating reset seeds would produce the same trajectory; this does not establish physical-drone performance or robustness to model error.

## Verification

- Gymnasium checks pass for the raw environment and normalized PPO wrapper.
- Recorded RNN calls verify input order, zero roll/pitch, normalization, denormalization, and acceleration integration.
- A skipped-climb landing receives failure; a completed climb/turn/soft landing qualifies; reset clears the recurrent state and altitude history.
- Notebook raw-action inference and main's wrapped inference produce bit-identical states, commands, and returns.
- Actual notebook setup, mission, helper, default Results, and reverse-turn Results cells execute successfully.
- Saved action transforms persist even when success-based checkpoint selection is disabled.
- Compatible PPO resume works; changed PPO settings or removed observation scaling are rejected.
- The existing altitude-only PPO training/MATLAB export path still passes a small regression check.
- Notebook syntax and `git diff --check` pass.

## Artifacts and use

- [Default flight](flight.png), [metrics](metrics.json), [trajectory CSV](trajectory.csv)
- [Initial training curve](initial_learning_curve.png), [continued training curve](learning_curve.png), [PPO losses](ppo_losses.png)
- [Resolved training config](training/ppo_config.json), [all mission metrics](mission_checks.json)
- Best policy: `training/best_success.zip`; final training state: `training/ppo_drone_rnn.zip`.
- Evaluate the installed policy: `python src/main_z_psi.py`.
- Train from scratch: `python src/main_z_psi.py --train` (1,500,000 steps, seed 1).
- Notebook: restart the kernel, run Setup, Load the RNN, Choose a mission, Plot helpers, then PPO — Results.

Value loss estimates critic prediction error; policy gradient loss is PPO's clipped optimization objective. Neither is a mission success score or expected to decrease monotonically. The trajectory, terminal outcome, tracking errors, and evaluation return establish whether the policy works.

## Dynamics identity

RNN file SHA256: `62df0ec2a39c48d027cc77331c30cd906aed6fe61417ece7af35e2deb196ea7e`.
Combined weights/normalization fingerprint: `b60c4e2b47d3413e34cdf3e2ddc0669795a8274e59a553153f135b4276c5c3e1`.
Input mean/std: `[-0.013671999797224998, 0.0, 0.0, 9.185700037050992e-05]` / `[1.6101000308990479, 0.0, 0.0, 0.11469999700784683]`.
Output mean/std: `[-10.0084, 0.0174]` / `[20.6563, 10.8079]`.
