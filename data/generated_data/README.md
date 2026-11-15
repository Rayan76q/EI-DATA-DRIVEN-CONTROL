Generated CSVs belong here. Supplied measurements remain in `../initial_data/`.

Run `generate_training_data` from `../data_generator/` with mode `z` or `z_psi`.
The generator samples every 0.05 s, including t=0, and writes one trajectory per
CSV row.

`<mode>_inputs.csv` interleaves [motor1, motor2, motor3, motor4].
`<mode>_outputs.csv` interleaves
[x, y, z, vx, vy, vz, ax, ay, az, phi, theta, psi, wx, wy, wz, wdotx, wdoty, wdotz].
`<mode>_dataset.json` records the period, horizon, schema and seed.

Train with `src/model/training/train.py --mode z|z_psi`, passing the two CSVs
and `--input-format motors --output-format plant`. The trainer converts motors
to controls and selects az, or [az, wdotz].

For the supplied scalar `bdd_in_mat_05.csv` / `bdd_out_mat_05.csv` pair, use
`--input-format controls --output-format selected`. New checkpoints save their
normalization and do not replace the supplied models.
