"""Inspect RNN acceleration predictions in physical units."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.rnn_loader import checkpoint_parameters, load_rnn_model
from model.training.train import read_csv
from policy.lib.droneEnv import PARAMS as Z_PARAMS
from policy.lib.droneEnvRotation import PARAMS as Z_PSI_PARAMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = {
    "z": ("rnn_z.pt", Z_PARAMS, ["u"], ["az"]),
    "z_psi": ("rnn_z_psi.pt", Z_PSI_PARAMS, ["thrust", "roll", "pitch", "yaw"], ["az", "wdotz"]),
}


def predict_accelerations(model, params, commands, rotation=False):
    """Normalize physical controls and carry recurrent state across the trajectory."""
    mean = np.asarray(params.mean_u, dtype=np.float32)
    scale = np.asarray(params.sigma_u, dtype=np.float32)
    # Roll and pitch are fixed at zero in the supplied altitude/yaw model.
    scale = np.where(scale == 0, 1, scale)
    normalized = (commands - mean) / scale
    output_mean = [params.mean_az]
    output_scale = [params.sigma_az]
    if rotation:
        output_mean.append(params.mean_wz_dot)
        output_scale.append(params.sigma_wz_dot)

    predictions = []
    hidden = None
    with torch.inference_mode():
        for command in normalized:
            output, hidden = model(torch.from_numpy(command).reshape(1, 1, -1), hidden)
            predictions.append(output.numpy().reshape(-1) * output_scale + output_mean)
    return np.asarray(predictions), normalized


def plot_predictions(times, commands, predictions, targets, input_names, output_names, path):
    """Plot physical inputs and predicted accelerations, with measurements when supplied."""
    fig, axes = plt.subplots(1 + len(output_names), 1, sharex=True, figsize=(10, 3 * (1 + len(output_names))))
    for index in ([0, 3] if len(input_names) == 4 else [0]):
        axes[0].step(times, commands[:, index], where="post", label=input_names[index])
    axes[0].set_ylabel("Physical control")
    for index, name in enumerate(output_names):
        axes[index + 1].plot(times, predictions[:, index], label="RNN")
        if targets is not None:
            axes[index + 1].plot(times, targets[:, index], "--", label="Measured")
        axes[index + 1].set_ylabel("az [m/s²]" if name == "az" else "wdotz [rad/s²]")
    for axis in axes:
        axis.grid(alpha=.3)
        axis.legend()
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Run constant controls or the first physical-control trajectory from a CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="z")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-csv", type=Path, help="One flattened control trajectory per row")
    parser.add_argument("--target-csv", type=Path, help="Matching az or [az, wdotz] samples")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--command", type=float, default=.394024)
    parser.add_argument("--yaw-command", type=float, default=0.)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.steps < 1 or not np.isfinite([args.command, args.yaw_command]).all():
        parser.error("Use a positive step count and finite commands")

    filename, defaults, input_names, output_names = MODES[args.mode]
    checkpoint = args.checkpoint or PROJECT_ROOT / "src/model/checkpoints" / filename
    model = load_rnn_model(checkpoint)
    params = checkpoint_parameters(model, defaults, args.mode)
    inputs, outputs = len(input_names), len(output_names)
    if (model.rnn.input_size, model.head.out_features) != (inputs, outputs):
        parser.error("Checkpoint dimensions do not match the selected mode")

    commands = np.zeros((args.steps, inputs), dtype=np.float32)
    commands[:, 0] = args.command
    if args.mode == "z_psi":
        commands[:, 3] = args.yaw_command
    if args.input_csv:
        row = read_csv(args.input_csv, max_rows=1)[0]
        if len(row) % inputs:
            parser.error(f"Input CSV requires {inputs} controls per sample")
        commands = row.reshape(-1, inputs)[:args.steps]

    targets = None
    if args.target_csv:
        row = read_csv(args.target_csv, max_rows=1)[0]
        if len(row) % outputs or len(row) < len(commands) * outputs:
            parser.error("Target CSV must contain matching acceleration samples")
        targets = row.reshape(-1, outputs)[:len(commands)]

    predictions, normalized = predict_accelerations(model, params, commands, args.mode == "z_psi")
    times = (np.arange(len(commands)) + 1) * params.dt
    columns = [times, *commands.T, *normalized.T, *predictions.T]
    header = ["time_s", *[f"{name}_physical" for name in input_names],
              *[f"{name}_standardized" for name in input_names],
              *[f"{name}_predicted" for name in output_names]]
    metrics = dict(mode=args.mode, checkpoint=str(checkpoint.resolve()),
                   samples=len(commands), dt_s=params.dt,
                   final_prediction=predictions[-1].tolist())
    if targets is not None:
        error = predictions - targets
        columns.extend([*targets.T, *error.T])
        header.extend([*[f"{name}_target" for name in output_names],
                       *[f"{name}_error" for name in output_names]])
        metrics.update(mae=np.abs(error).mean(axis=0).tolist(),
                       rmse=np.sqrt((error ** 2).mean(axis=0)).tolist())

    output_dir = args.output_dir or PROJECT_ROOT / "results/model" / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "rnn_inference.csv", np.column_stack(columns),
               delimiter=",", header=",".join(header), comments="")
    plot_predictions(times, commands, predictions, targets, input_names, output_names,
                     output_dir / "rnn_inference.png")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Predicted {len(commands)} samples with {checkpoint}; results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
