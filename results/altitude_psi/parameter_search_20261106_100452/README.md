# Rotation reward parameter search

Selected the lower-penalty Huber profile (`gentle`) and applied it as `REWARD_PARAMS` in `src/policy/lib/droneEnvRotation.py`. The existing reward formula remains recognizable; running rewards are now multiplied by dt=0.05. The RNN weights, normalization, raw input mapping, action limits, and dynamics function are unchanged.

The default mission and notebook were corrected to request a 2*pi-radian turn from 5 to 8 seconds and hold that yaw during landing. The earlier 360-radian reference also kept increasing during landing.

## Selected parameters

Values below are rates per second except the fixed terminal rewards:

| Parameter | Value |
|---|---:|
| alive_bonus | 2.0 |
| z_bonus | 4.0 |
| psi_bonus | 4.0 |
| z_penalty | 0.5 |
| psi_penalty | 0.2 |
| vz_penalty | 0.1 |
| wz_penalty | 0.02 |
| yaw_effort | 0.001 |
| smoothness | 0.005 |
| success | 50.0 |
| failure | 20.0 |

## Evidence and limits

All trials used the same frozen RNN and intended 10-second mission. PPO settings stayed at learning_rate=0.0001, gamma=0.99, n_steps=256, n_envs=10, with the shared configuration unchanged. Staged continuations reset the random generator when the checkpoint was loaded. The `damped` trial branched from the gentle policy at 501,760 steps; it was not trained from scratch.

| Reward profile / seed | Total trained steps | Best saved policy survives | Altitude MAE | Yaw MAE | Outcome |
|---|---:|---:|---:|---:|---|
| Original / 0 | 501,760 | 0.30 s | 0.189 m | 0.015 rad | Crashed |
| Selected / 0 | 1,502,720 | 10.00 s | 0.633 m | 1.470 rad | Timeout |
| Selected / 1 | 501,760 | 10.00 s | 0.590 m | 1.482 rad | Timeout |
| Bounded / 0 | 1,502,720 | 10.00 s | 0.600 m | 1.369 rad | Timeout |
| Stronger damping / 0 | 1,502,720 | 10.00 s | 0.713 m | 1.333 rad | Timeout, yaw target reached |

The selected profile escaped the startup crash in both tested seeds. No tested PPO profile reliably completed the full turn plus safe landing. The bounded profile had a competitive early checkpoint but strongly degraded later; the selected profile keeps the simpler existing Huber penalties and was checked with a second seed. These are improved starting parameters, not an optimal or mission-solved result. Short crash trajectories have deceptively low MAE because they never reach the harder phases; inspect duration and phase coverage alongside MAE.

A separate planned raw-model feasibility trajectory landed and turned. The old reward scored it -5797.69, below an early crash (-988.13); the selected reward scores it +104.02, above an early crash (-18.03). This checks reward ordering, not physical accuracy of the RNN.

The notebook heuristic is a simple manual input/output test: up, neutral, positive yaw, down, negative yaw, neutral. It does not optimize commands or try to solve the mission. `heuristic_io/input_output.png` shows the exact input channels and denormalized acceleration responses. Neutral thrust does not cancel existing velocity.

## Artifacts

- `comparison.csv`, `comparison.json`, and `comparison.png`: checkpoint comparison.
- `<profile>_seed<seed>/`: parameters, monitor data, evaluations, final and best policies.
- `selected_policy/ppo_drone_thrust_speeds.zip`: selected seed-1 checkpoint, 380,000 steps.
- `selected_policy/parameters.json`: its parameters and limitations.
- `selected_policy/evaluation/`: standard main_z_psi plot, CSV, NPZ, and metrics.
- `reward_order_check.json`: actual raw-model feasible flight versus early crash.
- `experiment.json`: RNN hash, normalization, input order and mission.
- `search_reward.py`: search runner; resume arguments reproduce staged continuations.

Use `python src/main_z_psi.py --train` to train with the new defaults. To inspect the tested policy, pass `--policy results/altitude_psi/parameter_search_20261106_100452/selected_policy/ppo_drone_thrust_speeds.zip`.
