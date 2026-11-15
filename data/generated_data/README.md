Generated CSVs go here; supplied measurements stay in `../initial_data/`.
MATLAB generation scripts are in `../data_generator/`, and the canonical plant
is in `../../src/drone/plant/`. CSVs are ignored by Git.

From the repository root in MATLAB:

```matlab
addpath('data/data_generator');
generate_training_data('z', fullfile(pwd,'data','generated_data'), 50, 8, 0);
generate_training_data('z_psi', fullfile(pwd,'data','generated_data'), 50, 8, 0);
```

Each CSV row is a complete trajectory sampled every 0.05 s, including t=0.
`<mode>_inputs.csv` interleaves four motor commands per sample;
`<mode>_outputs.csv` interleaves
`[x,y,z,vx,vy,vz,ax,ay,az,phi,theta,psi,wx,wy,wz,wdotx,wdoty,wdotz]`.
`<mode>_dataset.json` records the schema, period, horizon and random seed.
MATLAB's empty trailing CSV column is accepted by the shared Python parser.

Train either mode with `src/model/training/train.py --mode z|z_psi`, passing
`--inputs`, `--outputs`, `--input-format motors --output-format plant`.
The trainer selects az (and wdotz), preserves trajectory boundaries, and saves
training normalization with the weights. New model weights remain separate
from the existing RL-compatible checkpoints.

The supplied scalar `bdd_in_mat_05.csv` / `bdd_out_mat_05.csv` pair can also
train z with `--input-format controls --output-format selected`. The older
`rnn_input_z.csv` / `rnn_output_z.csv` layout is supported through
`--output-format z_state`; it is no longer a required missing dataset.
