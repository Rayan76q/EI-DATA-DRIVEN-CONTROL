# Data-driven drone control

RNN dynamics, reinforcement learning and Simulink deployment for the
[EINT project](EINT_student_version_EN.pdf). Two missions are supported at a
0.05 s control period: climb/hover/land on z, and climb/hover/complete a full
turn on ψ/land. The existing RL configurations and policy weights are preserved.

## Run the saved policies

From the repository root, install `requirements.txt` in your Python environment:

```sh
python -m pip install -r requirements.txt
python src/main_z.py --output-dir results/altitude/demo
python src/main_z_psi.py --output-dir results/altitude_psi/demo
```

Both mains evaluate the saved policy on its RNN, export it, check its numerical
agreement in MATLAB, and run the actual Simulink plant. MATLAB is discovered
through `--matlab PATH`, `MATLAB_EXECUTABLE`, `PATH`, or `~/bin/matlab`.
Use `--skip-matlab` for Python evaluation and export only. Missing MATLAB or
Simulink is recorded explicitly; it never counts as a plant pass. MATLAB logs,
status, physical trajectories and plots are saved beside the Python results.

The altitude mission requests 0 m until 2 s, 2 m until 6 s, then −0.3 m, with a
10 s horizon. The z+ψ mission uses the schedule in
`policy.lib.droneEnvRotation.default_mission`: a 2 m hover, an unwrapped 2π turn,
and landing, with a 20 s horizon. The policies observe current tracking errors
and motion; they do not observe the future schedule.

| Main | Observation | Physical control | Default dynamics |
| --- | --- | --- | --- |
| `src/main_z.py` | z−z_ref, vz, az | Collective motor command | `rnn_z.pt`, 1 input / 1 acceleration output |
| `src/main_z_psi.py` | z−z_ref, vz, az, ψ−ψ_ref, wz, wdotz | Thrust and differential yaw | `rnn_model_thrust_speeds.pt`, 4 controls / 2 acceleration outputs |

The altitude main accepts `--rl-model ppo|dqn|sac`, `--policy PATH`, `--rnn PATH`
and `--config PATH`. DQN requires its `commands.npy` beside the checkpoint.
The rotation main accepts `--policy`, `--rnn` and `--config` too; its current
saved policy is PPO. Its saved observation and action scaling are restored
from the agent, and a dynamics fingerprint rejects an incompatible RNN.

## Integration results

The integration on 2026-11-15 ran MATLAB R2026b Update 1 and Simulink with zero
plant noise. See [the validation report](results/integration/20261115/report.json)
and [the detailed findings](results/integration/20261115/README.md).

| Saved policy | Simulink mission | First touchdown | Vertical speed |
| --- | --- | --- | --- |
| PPO z, main's selected saved run | Passed | 8.00 s | +0.009 m/s |
| PPO z+ψ | Passed; yaw error about 1.39° | 11.75 s | −0.074 m/s |
| DQN z | Passed | 7.30 s | −0.261 m/s |
| SAC z, selected 389,120-step checkpoint | Failed: hard landing | 7.00 s | −1.321 m/s |

The bundled `ppo_drone_rnn.zip` was also checked: it lands at 8.20 s with
−0.125 m/s vertical speed. The report records each exact checkpoint path.

SAC's exported actor agrees numerically with Stable-Baselines3, and the saved
policy lands on its RNN dynamics. Its plant transfer does not meet the landing
speed limit. The older 327,680-step checkpoint also fails on the plant
(−1.403 m/s); another older policy reaches −0.508 m/s, still outside the
0.5 m/s limit. As requested, the RL policies and configurations remain intact;
this integration does not retune SAC or replace its checkpoint.

Plant acceptance is measured at the first sampled z≤−0.3 m during landing,
with |vz|≤0.5 m/s and at least 0.5 s cumulative occupancy within ±0.2 m of the
positive altitude reference before touchdown. Rotation additionally requires
unwrapped yaw error≤10° and |wz|≤0.5 rad/s. This occupancy measure does not
assert a continuous settled hover. The plant has no contact model and continues
to the full horizon after this virtual touchdown; later samples do not change
the first-touchdown verdict. These checks establish the reported deterministic
missions, rather than robustness across noise and initial conditions.

## Generate data and train the RNN

The standalone `EINT_system_simulation.slx` now reads `sim_u1`…`sim_u4` from
workspace sources. Previously its motors were driven by fixed demonstration
steps, so the generators' random inputs had no effect. Both the canonical
model and the team's `matlab_files/` copy have been repaired. UAV physics is
unchanged. The PID model needed from `origin/main` is copied into
`src/drone/plant/EINT_system_simulationPID.slx`; the `main` branch is unchanged.

In MATLAB, from the repository root:

```matlab
addpath('data/data_generator');
generate_training_data('z', fullfile(pwd,'data','generated_data'), 50, 8, 0);
generate_training_data('z_psi', fullfile(pwd,'data','generated_data'), 50, 8, 0);
```

The generator uses Ts=0.05 s and writes headerless CSVs with one trajectory per
row: four interleaved motor commands and all 18 interleaved physical states.
A JSON file records the schema. It checks that different episodes actually
excite acceleration and, for z+ψ, yaw acceleration. Legacy scalar generators
also provide all four workspace signals at the control period.

```sh
python src/model/training/train.py --mode z \
  --inputs data/generated_data/z_inputs.csv \
  --outputs data/generated_data/z_outputs.csv \
  --input-format motors --output-format plant --onnx
python src/model/training/train.py --mode z_psi \
  --inputs data/generated_data/z_psi_inputs.csv \
  --outputs data/generated_data/z_psi_outputs.csv \
  --input-format motors --output-format plant --onnx
```

The callable trainer preserves trajectory boundaries, splits training/test
trajectories, and saves normalization learned from training data in `rnn.pt`.
The z model has a 32-unit tanh RNN and linear az head; z+ψ has 64 units and a
linear [az, wdotz] head. Four motors are converted to
[thrust, roll, pitch, yaw] using the team's mixer. Scalar z training requires
equal motor commands. `--input-format controls` accepts already converted
controls, and `--output-format selected` accepts acceleration outputs directly.
The supplied `bdd_in_mat_05.csv` / `bdd_out_mat_05.csv` pair can train z using
these defaults. `--output-format z_state` accepts interleaved [z,vz,az].

Use `--epochs`, `--batch-size`, `--device cpu|cuda`, `--seed`, `--output-dir`
and `--max-trajectories` to control a run. The optional ONNX export has explicit
hidden-state input/output and dynamic batch/sequence dimensions. Existing
policy-compatible checkpoints are not replaced automatically. New bundles load
through the shared loader and supply their normalization to the environments;
new dynamics require policies trained for those dynamics.

`src/model/training/rnn_training.py` and `rnn_training_z_psi.py` are CLI wrappers.
To inspect saved-model inference:

```sh
python src/main_model.py --mode z
python src/main_model.py --mode z_psi --yaw-command 0.05
python src/main_model.py --mode z --checkpoint results/model_training/<run>/rnn.pt
```

The data-cleaning and RNN inspection notebooks use portable repository paths,
shared CSV parsing and checkpoint normalization. They support the actual
provided dataset sizes rather than assuming 1,000 samples per trajectory.

## Train RL

The mains and RL notebooks use the shared JSON configs and training functions:

```sh
python src/policy/training.py --policy ppo
python src/policy/training.py --policy dqn
python src/policy/training.py --policy sac
python src/main_z_psi.py --train
```

`--steps` and `--seed` override the selected config for that run; resolved
settings, checkpoint selection and TensorBoard events are recorded. Training
runs the requested budget, rounded to the algorithm's collection block, without
reward-threshold stopping. It saves both the final agent and the selected agent;
later training can regress. Evaluation frequency is expressed in total
transitions and divided by the environment count for the SB3 callback.

The preserved settings are PPO z 700,000 steps / 10 environments, PPO z+ψ
1,500,000 / 8, DQN 500,000 / 10, and SAC 500,000 / 4. DQN retains its successful
n_steps=3, tau=0.1, train_freq=5 and gradient_steps=−1. SAC retains
train_freq=4 and gradient_steps=4. The saved SAC continuation begins at the
327,680-step checkpoint with a new replay buffer and a 2,000-transition warmup;
its configured replay checkpoint is the selected 389,120-step agent.
Use `--from-scratch` with SAC training for an independent comparison. These are
algorithm-specific settings, not a claim of equal optimization budgets.

The rotation policy saves its observation divisors and affine action mapping.
Python and MATLAB apply exactly those transformations. The raw motor mixer is
[thrust+yaw, thrust−yaw, thrust+yaw, thrust−yaw]. Yaw is unwrapped so that zero
heading does not count as completing a full turn.

To view training logs:

```sh
python -m tensorboard.main --logdir results --port 6006
```

## MATLAB exports and checks

Each main writes `policy.mat`, `traj_model.mat`, and `rnn.mat` alongside its
Python plots, trajectory and metrics. The policy file contains weights,
activations, observation scaling, physical action scaling/clipping, mission
samples, and independent golden observations/actions. DQN exports its command
grid; SAC exports its deterministic mean action. The RNN export includes its
recurrence, linear head and physical normalization with golden predictions.

Run manually from the repository root:

```matlab
addpath('src/drone/closed_loop');
folder = fullfile(pwd,'results','altitude_psi','demo');
check_policy_export(folder);
check_rnn_export(folder);
EINT_RL_eval2(folder, 0.0); % six observations / two controls
% EINT_RL_eval(folder, 0.0) selects either mission from policy.mat.
```

`EINT_RL_eval` selects `EINT_RL_closed_loop.slx` or
`EINT_RL_closed_loop_2.slx`. It saves all 18 plant states, references and motor
commands in `trajectory.csv` and `traj_simulink.mat`, plus altitude, yaw and
state figures. A third argument chooses the output directory; a fourth `false`
skips figures. It does not save changes to the closed-loop source models.
The mains also write `matlab_status.json`, `run_matlab.m` and `matlab.log`.
A completed simulation that violates touchdown criteria is `mission_failed`.

## Layout and validation

`src/model/` contains dynamics loading/training/export and model notebooks;
`src/policy/` contains shared configs, training, evaluation/export and RL
notebooks. `src/drone/plant/` is the canonical plant location;
`src/drone/closed_loop/` contains its deployment adapters and numerical checks.
The incoming team's `matlab_files/`, `simulation_3d/` and notebooks remain
available. `data/initial_data/` holds supplied measurements, while generated
CSVs and run artifacts go in `data/generated_data/` and `results/`.
Tensor storage directories inside extracted PyTorch checkpoints are not CSV
training datasets and must stay beside `data.pkl`, `version` and `byteorder`.

Run the interface regression suite with:

```sh
python -m unittest discover -s tests -p 'test_*.py'
```

It covers recurrent state, motor/output layouts, saved scaling, full-turn
semantics, training metadata and complete rotation export. Integration evidence
also includes one-epoch RNN training in both modes, short PPO/DQN/SAC training
runs into separate output directories, notebook execution, ONNX parity and
actual MATLAB/Simulink runs. Short training runs check execution and contracts;
they do not establish convergence of newly trained models.

Existing estimator tests:
`python -m unittest discover -s src/model/estimation -p test_rnn_ekf.py`.
The optional viewer needs Meshcat:
`PYTHONPATH=src python -m drone.visualization.demo`.
