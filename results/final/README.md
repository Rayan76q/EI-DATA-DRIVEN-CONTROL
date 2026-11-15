# Reference runs

`z/` contains the selected altitude PPO policy. `z_psi/` contains the selected
PPO policy for altitude landing and a full yaw turn. These are the same policy
weights used before the source cleanup. Historical training runs remain in the
other results folders.

Each folder contains `policy.zip`, the original `training_config.json` and
`learning_curve.png`. The `rnn/` folder shows acceleration predictions on a
sample of supplied data. It is an inspection, not a held-out accuracy estimate.
The two RNN checkpoints remain in `src/model/checkpoints/`.

`flight.png`, `trajectory.csv` and `metrics.json` describe the RNN flight.
`policy.mat`, `rnn.mat` and `traj_model.mat` are the complete MATLAB export.
`plant/` contains the physical trajectory and its flight plot. The MATLAB
batch script, short log and numerical check results are included.

Both plant missions pass with zero noise. Altitude touchdown is at 8.00 s;
altitude plus yaw touchdown is at 11.75 s with about 1.39° yaw error.
`validation.json` records the checks after cleanup. SAC remains available as a
comparison in its original results folder; its plant landing speed still
exceeds the 0.5 m/s limit.

To regenerate either reference run from the repository root:

```sh
python src/main_z.py --output-dir results/final/z
python src/main_z_psi.py --output-dir results/final/z_psi
```
