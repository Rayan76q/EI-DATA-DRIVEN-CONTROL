"""Run the altitude RNN on a constant command or the first row of a CSV.

Inputs are physical motor commands; targets are vertical acceleration in m/s².
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from model.rnn_loader import load_rnn_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Original altitude-model training statistics; normalize commands exactly once.
INPUT_MEAN = -0.004098882433027029
INPUT_STD = 1.7267645597457886
OUTPUT_MEAN = -9.68746566772461
OUTPUT_STD = 21.440523147583008
DT = 0.05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "src/model/checkpoints/rnn_z.pt",
    )
    parser.add_argument("--input-csv", type=Path, help="Physical commands in the first CSV row")
    parser.add_argument("--target-csv", type=Path, help="Physical az targets in the first CSV row")
    parser.add_argument("--steps", type=int, default=200, help="Maximum number of samples")
    parser.add_argument("--command", type=float, default=0.394024, help="Constant command without --input-csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/model")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not np.isfinite(args.command):
        parser.error("--command must be finite")

    commands = np.full(args.steps, args.command, dtype=np.float64)
    targets = None
    for argument, label in ((args.input_csv, "input"), (args.target_csv, "target")):
        if argument is None:
            continue
        with argument.open(newline="") as handle:
            row = next(csv.reader(handle), [])
        # MATLAB generators may leave a trailing comma after the final sample.
        while row and not row[-1].strip():
            row.pop()
        try:
            samples = np.asarray(row, dtype=np.float64)
        except ValueError:
            parser.error(f"{label} CSV first row must contain numeric samples")
        if samples.size == 0 or not np.isfinite(samples).all():
            parser.error(f"{label} CSV first row must contain finite samples")
        if label == "input":
            commands = samples[:args.steps]
        else:
            if len(samples) < len(commands):
                parser.error("target CSV has fewer samples than the command trajectory")
            targets = samples[:len(commands)]

    model = load_rnn_model(args.checkpoint)
    if (model.rnn.input_size != 1 or not isinstance(model.head, nn.Linear)
            or model.head.out_features != 1):
        parser.error("this altitude entrypoint requires a scalar-input, scalar-output RNN")
    normalized_commands = (commands - INPUT_MEAN) / INPUT_STD
    predictions = np.empty(len(commands), dtype=np.float64)
    hidden = None
    with torch.inference_mode():
        for index, command in enumerate(normalized_commands):
            tensor = torch.tensor([[[command]]], dtype=torch.float32)
            output, hidden = model(tensor, hidden)
            predictions[index] = output.item() * OUTPUT_STD + OUTPUT_MEAN

    times = (np.arange(len(commands)) + 1) * DT
    columns = [times, commands, normalized_commands, predictions]
    header = "time_s,u_physical,u_standardized,az_predicted_m_s2"
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "input_csv": str(args.input_csv.resolve()) if args.input_csv else None,
        "target_csv": str(args.target_csv.resolve()) if args.target_csv else None,
        "samples": len(commands), "dt_s": DT,
        "input_size": model.rnn.input_size, "hidden_size": model.rnn.hidden_size,
        "num_layers": model.rnn.num_layers,
        "input_mean": INPUT_MEAN, "input_std": INPUT_STD,
        "output_mean": OUTPUT_MEAN, "output_std": OUTPUT_STD,
        "final_az_m_s2": float(predictions[-1]),
        "min_az_m_s2": float(predictions.min()),
        "max_az_m_s2": float(predictions.max()),
    }
    if targets is not None:
        error = predictions - targets
        columns.extend([targets, error])
        header += ",az_target_m_s2,az_error_m_s2"
        metadata.update(mae_m_s2=float(np.abs(error).mean()),
                        rmse_m_s2=float(np.sqrt(np.mean(error ** 2))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "rnn_inference.csv"
    np.savetxt(csv_path, np.column_stack(columns), delimiter=",", header=header, comments="")
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].step(times, commands, where="post", label="Physical command")
    axes[0].set_ylabel("Motor command")
    axes[1].plot(times, predictions, label="RNN prediction")
    if targets is not None:
        axes[1].plot(times, targets, "--", label="Measured target")
    axes[1].set_ylabel("Vertical acceleration (m/s²)")
    axes[1].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Altitude dynamics RNN")
    figure.tight_layout()
    figure.savefig(args.output_dir / "rnn_inference.png", dpi=150)
    plt.close(figure)
    (args.output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Predicted {len(commands)} samples; final az = {predictions[-1]:.6f} m/s²")
    if targets is not None:
        print(f"MAE = {metadata['mae_m_s2']:.6f}; RMSE = {metadata['rmse_m_s2']:.6f} m/s²")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
