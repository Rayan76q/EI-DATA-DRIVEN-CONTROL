Place generated training CSVs here. Generation scripts are in
`../data_generator/`; Simulink models are in `../../src/drone/plant/`.
Keep supplied measurements in `../initial_data/` and model weights under
`../../src/model/checkpoints/`. CSV files are ignored by Git.

The altitude trainer expects `rnn_input_z.csv` and `rnn_output_z.csv`, with
one trajectory per row: 1,000 commands and 3,000 interleaved `(z, vz, az)`
values. The supplied initial data has 2,000 samples per row and
acceleration-only outputs. It cannot replace this training format.
The revised dataset and training code still need to be supplied.
