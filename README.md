# Drone control

Two control tasks: climb, hover and land on z; or do the same while completing
a full turn on ψ. Both use an RNN dynamics model and a policy deployed to the
Simulink plant. The control period is 0.05 s.

## Run

```sh
python -m pip install -r requirements.txt
python src/main_z.py
python src/main_z_psi.py
```

Each command evaluates the saved policy on its RNN, exports it to MATLAB and
runs the plant. MATLAB is found through `--matlab PATH`, `MATLAB_EXECUTABLE`,
`PATH` or `~/bin/matlab`. Use `--skip-matlab` for Python only.

Selected policies and compact reference results are in
[`results/final/z`](results/final/z) and
[`results/final/z_psi`](results/final/z_psi). New runs go to `results/runs/`.
The JSON configs in `src/policy/config/` select the saved policies and training
settings. `--policy PATH`, `--rnn PATH` and `--config PATH` override them.

To evaluate DQN or SAC:

```sh
python src/main_z.py --rl-model dqn
python src/main_z.py --rl-model sac
```

PPO passes both plant missions. DQN also soft-lands. The preserved SAC policy
lands on the RNN but reaches the plant at about 1.32 m/s, above the 0.5 m/s
landing limit. A plant failure is reported as `mission_failed`.

## Data and RNN training

The full pipeline is Simulink data generation, RNN training, policy training,
then closed-loop Simulink evaluation.

Generate data from MATLAB, starting at the repository root:

```matlab
addpath('data/data_generator');
generate_training_data('z', fullfile(pwd,'data','generated_data'), 50, 8, 0);
generate_training_data('z_psi', fullfile(pwd,'data','generated_data'), 50, 8, 0);
```

Each CSV row contains one trajectory. Inputs interleave four motor commands;
outputs interleave all 18 plant states. The generator writes a JSON schema
beside the CSVs and checks that motor commands excite the expected motion.

```sh
python src/model/training/train.py --mode z \
  --inputs data/generated_data/z_inputs.csv \
  --outputs data/generated_data/z_outputs.csv \
  --input-format motors --output-format plant
python src/model/training/train.py --mode z_psi \
  --inputs data/generated_data/z_psi_inputs.csv \
  --outputs data/generated_data/z_psi_outputs.csv \
  --input-format motors --output-format plant
```

Training saves `rnn.pt` and `training.json` under `results/model_training/`.
The checkpoint includes normalization learned from the training trajectories.
Use `--epochs`, `--output-dir` and `--device cpu|cuda` to control a run.
New dynamics need a policy trained for those dynamics.

Only two supplied RNN checkpoints remain in `src/model/checkpoints/`:
`rnn_z.pt` predicts az from collective control; `rnn_z_psi.pt` predicts
[az, wdotz] from [thrust, roll, pitch, yaw]. Their physical normalization is in
the corresponding environment. New training bundles carry their own statistics.

```sh
python src/main_model.py --mode z
python src/main_model.py --mode z_psi --yaw-command 0.05
```

`src/model/notebooks/` contains data preparation and RNN inspection.
`src/policy/notebooks/` contains one small notebook per mission. They call the
same mains and import the same configs as the command line.

## Policy training

```sh
python src/main_z.py --train
python src/main_z.py --train --rl-model dqn
python src/main_z.py --train --rl-model sac
python src/main_z_psi.py --train
```

`--steps` and `--seed` override the config for that run. Training saves the
selected and final checkpoints, resolved settings and TensorBoard logs. It
runs the full budget without reward-threshold stopping. SAC starts from its
configured checkpoint; `--from-scratch` disables that starting point.

The saved rotation policy includes observation scaling and the physical action
mapping. Python and MATLAB use the same mapping. Yaw stays unwrapped so zero
heading cannot count as a completed turn. Do not change scaling or RNN weights
when replaying a policy trained with them.

## MATLAB and checks

`src/drone/plant/` holds the plant. `src/drone/closed_loop/` holds the two
controller diagrams, inference functions and numerical checks. There is one
copy of each model. MATLAB uses `policy.mat`, `rnn.mat` and `traj_model.mat`
written by the mains; ONNX is not used.

For a manual replay:

```matlab
addpath('src/drone/closed_loop');
EINT_RL_eval(fullfile(pwd,'results','final','z'), 0.0);
EINT_RL_eval(fullfile(pwd,'results','final','z_psi'), 0.0);
```

Results include Python and plant trajectories, plots, metrics and MATLAB logs.
Success is measured at first touchdown. The plant continues after touchdown
because it has no contact model. Reference runs use zero plant noise.

The earlier RL training results remain under `results/altitude/`,
`results/altitude_psi/` and `results/policy/`. Old source-folder logs and full
experiment notebooks are archived there too. The optional 3D viewer has one
copy in `src/drone/visualization/` and requires Meshcat.
