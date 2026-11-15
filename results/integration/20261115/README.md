# Integration validation — 15 November 2026

The team's `origin/final` changes are merged into `final`, with the existing
local RL work preserved. `main` remains at
`d594c94d9957a4a1a7534402b2b8a3008ceca04f`.
The missing PID model was read from `origin/main` and copied into the canonical
plant directory without checking out or modifying `main`.

[report.json](report.json) records checkpoint identities, numerical checks,
physical mission results, training checks and limitations.
[preservation.json](preservation.json) records hashes for the preserved configs,
checkpoint files and altitude notebook. The preparation and merge commits are
`9ff94d9` and `a505ddf`.

## Physical results

MATLAB R2026b Update 1 and the installed Simulink runtime executed the actual
plant. Both main commands returned success after checking the actor and RNN
numerically and flying the mission:

```sh
python src/main_z.py --output-dir results/integration/20261115/main_z
python src/main_z_psi.py --output-dir results/integration/20261115/main_z_psi
```

| Policy | First touchdown | vz | Result |
| --- | --- | --- | --- |
| Altitude main's selected PPO run | 8.00 s | +0.0088 m/s | Pass |
| Bundled altitude PPO | 8.20 s | −0.1249 m/s | Pass |
| Altitude + full-turn PPO | 11.75 s | −0.0740 m/s | Pass |
| Restored DQN | 7.30 s | −0.2609 m/s | Pass |
| Selected SAC, 389,120 steps | 7.00 s | −1.3209 m/s | Fail |

At rotation touchdown the plant reaches ψ=6.25887 rad, an unwrapped yaw error
of −0.02432 rad (−1.39°), with yaw rate 0.00893 rad/s. It also meets the altitude
occupancy requirement. Both primary missions use the existing saved policies.

The SAC policy lands on the RNN at 7.55 s with vz=−0.4305 m/s. Its exported
actor agrees with Stable-Baselines3 in MATLAB, so the observed plant hard
landing is a transfer limitation. Two older SAC checkpoints were also tested:
`20261105_202000_ztol02` lands at −1.4033 m/s and
`20261105_193731_full` at −0.5079 m/s. Neither meets the 0.5 m/s limit; the
threshold was not relaxed. The user explicitly chose to preserve the policies
and document this limitation rather than tune SAC during integration.

Verdicts use the first 0.05 s sample at/below −0.3 m during the landing phase,
|vz|≤0.5 m/s, and at least 0.5 s cumulative occupancy within ±0.2 m of the
positive altitude reference. Rotation adds unwrapped yaw error≤10° and
|wz|≤0.5 rad/s. The plant runs to its full horizon after virtual touchdown;
those later samples cannot rescue a failed first contact. There is no physical
contact model. Only nominal zero-noise missions were tested.

The local run folders contain physical CSVs, raw solver-time MAT files, scope
figures, MATLAB logs and the golden export data. Selected summary reports and
flight figures are versioned; large generated outputs and caches remain local.

## Repairs and execution checks

- Export and deployment now support z's 3 observations / 1 action and z+ψ's
  6 observations / 2 actions. Saved observation scaling, physical action mapping,
  yaw reference, acceleration feedback and motor mixing agree across Python
  and MATLAB. Golden tests cover independent observations as well as replays.
- MATLAB RNN recurrence/normalization checks pass in both modes: maximum
  prediction errors are about 2.9e−6 and 1.6e−5. All four actor variants pass
  MATLAB agreement checks; DQN's discrete actions retain their physical grid.
- Shared RNN training supports scalar and four-control models, preserves whole
  trajectories, records normalization with its weights, and exports explicit
  recurrent state to ONNX. One-epoch checks ran on eight supplied trajectories
  in each mode; these check execution, not newly trained model convergence.
- Both standalone generation models originally used fixed Step/Step1 demo
  inputs instead of the scripts' motor signals. They now connect workspace
  `sim_u1`…`sim_u4`, with sample-and-hold at Ts=0.05 s. The protected physics
  and UAV subsystem files are unchanged. Source models remain R2024a compatible.
- Fresh Simulink-generated z and z+ψ datasets, each with three seven-sample
  trajectories, pass excitation assertions, training, checkpoint reload and
  ONNX/PyTorch parity. Maximum ONNX errors are below 1.5e−7.
- Short PPO z, DQN z, SAC z and PPO z+ψ training runs complete and export into
  separate validation folders. Original policy checkpoints are not replaced.
- The three RNN notebooks' loading/preparation/inference/export cells executed
  on reduced data. All six notebooks validate and compile. ONNX tests cover
  the existing scalar model's streaming interface and dynamic exports in both
  modes; the historical scalar ONNX file remains one-step by design.
- Eight interface regression tests pass, including first-contact landing
  evaluation and the distinction between zero yaw and a complete turn.

To repeat the interface tests:

```sh
python -m unittest discover -s tests -p 'test_*.py'
```

Data generation and training commands for either mode are in the root README.
`collect_report.py` rebuilds the summary from the local integration artifacts;
it does not rerun simulations or training. `run_plant_checks.m`,
`run_remaining_checks.m`, and `run_sac_transfer.m` record the manual comparison
runs, while each main writes its own reproducible `run_matlab.m`.
