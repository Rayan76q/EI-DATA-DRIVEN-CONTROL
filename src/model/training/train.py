"""Train the altitude or altitude/yaw dynamics RNN from complete trajectories.

Run ``python src/model/training/train.py --help`` for the CSV layouts. Each run
saves its weights, normalization and train/test split in a separate directory.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MOTOR_TO_CONTROL = np.array(
    [
        [0.25, 0.25, 0.25, 0.25],
        [0, 0.5, 0, -0.5],
        [-0.5, 0, 0.5, 0],
        [0.25, -0.25, 0.25, -0.25],
    ],
    dtype=np.float32,
)


def read_csv(path, max_rows=None):
    """Read one flattened trajectory per row, allowing empty trailing columns."""
    values = np.genfromtxt(path, delimiter=",", dtype=np.float32, ndmin=2, max_rows=max_rows)
    while values.shape[1] and np.isnan(values[:, -1]).all():
        values = values[:, :-1]
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Dataset must contain finite numeric trajectories: {path}")
    return values


def prepare_sequences(inputs, outputs, mode, input_format="controls", output_format="selected"):
    """Reshape CSV rows to trajectories and select [az] or [az, wdotz].

    Controls are [thrust] for z or [thrust, roll, pitch, yaw] for z_psi. Raw motor
    inputs are converted to that order. Plant outputs contain 18 states per step;
    selected outputs already contain only the acceleration channels being learned.
    """
    if mode not in {"z", "z_psi"}:
        raise ValueError("Mode must be z or z_psi")
    if input_format not in {"controls", "motors"} or output_format not in {"selected", "plant"}:
        raise ValueError("Choose controls/motors inputs and selected/plant outputs")

    rotation = mode == "z_psi"
    input_width = 4 if input_format == "motors" or rotation else 1
    output_width = 18 if output_format == "plant" else (2 if rotation else 1)
    if inputs.ndim != 2 or outputs.ndim != 2 or inputs.shape[0] != outputs.shape[0]:
        raise ValueError("Input and output CSVs must have the same trajectory count")
    if inputs.shape[1] % input_width or outputs.shape[1] % output_width:
        raise ValueError("CSV widths do not match the declared input/output layouts")

    x = inputs.reshape(len(inputs), -1, input_width)
    y = outputs.reshape(len(outputs), -1, output_width)
    if x.shape[1] != y.shape[1]:
        raise ValueError("Input and output trajectories have different sample counts")

    if input_format == "motors":
        controls = x @ MOTOR_TO_CONTROL.T
        if not rotation and not np.allclose(controls[..., 1:], 0, atol=1e-5):
            raise ValueError("Scalar z training requires equal motor commands; use z_psi for yaw data")
        x = controls if rotation else controls[..., :1]
    if output_format == "plant":
        y = y[..., [8, 17] if rotation else [8]]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Training sequences contain non-finite values")
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class DynamicsRNN(nn.Module):
    """A tanh RNN with a linear acceleration output head."""

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        output, hidden = self.rnn(x, hidden)
        return self.head(output), hidden


def train(
    inputs,
    outputs,
    output_dir,
    *,
    mode="z",
    input_format="controls",
    output_format="selected",
    epochs=700,
    batch_size=32,
    seed=0,
    hidden_size=None,
    learning_rate=1e-3,
    max_rows=None,
    dt=0.05,
    device=None,
):
    """Train with a 90/10 split by trajectory and save a portable CPU bundle.

    Normalization uses only the training split. The saved statistics accompany
    the weights so inference uses exactly the same units as this training run.
    Returns the evaluation model and the path to its checkpoint.
    """
    if mode not in {"z", "z_psi"} or epochs <= 0 or batch_size <= 0 or dt <= 0:
        raise ValueError("Choose a supported mode and positive training settings")

    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(seed)
    x, y = prepare_sequences(
        read_csv(inputs, max_rows), read_csv(outputs, max_rows), mode, input_format, output_format
    )
    if len(x) < 3:
        raise ValueError("At least three complete trajectories are required for train/test splitting")

    order = rng.permutation(len(x))
    split = min(len(x) - 1, max(1, int(0.9 * len(x))))
    train_indices, test_indices = order[:split], order[split:]
    input_mean = x[train_indices].mean(axis=(0, 1))
    input_scale = x[train_indices].std(axis=(0, 1)) + 1e-5
    output_mean = y[train_indices].mean(axis=(0, 1))
    output_scale = y[train_indices].std(axis=(0, 1)) + 1e-5
    normalized_inputs = (x - input_mean) / input_scale
    normalized_outputs = (y - output_mean) / output_scale

    dataset = TensorDataset(
        torch.from_numpy(normalized_inputs[train_indices]),
        torch.from_numpy(normalized_outputs[train_indices]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    hidden_size = hidden_size or (64 if mode == "z_psi" else 32)
    model = DynamicsRNN(x.shape[-1], y.shape[-1], hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_updates = epochs * len(loader)
    warmup = max(1, int(0.1 * total_updates))

    def learning_rate_factor(step):
        """Warm up for 10% of updates, then decay with a cosine schedule."""
        if step < warmup:
            return 0.001 + 0.999 * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    test_inputs = torch.from_numpy(normalized_inputs[test_indices]).to(device)
    test_outputs = torch.from_numpy(normalized_outputs[test_indices]).to(device)
    criterion = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_inputs, batch_outputs in loader:
            batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
            optimizer.zero_grad()
            prediction, _ = model(batch_inputs)
            loss = criterion(prediction, batch_outputs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            prediction, _ = model(test_inputs)
            test_loss = float(criterion(prediction, test_outputs))
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss})
        print(
            f"Epoch {epoch + 1}/{epochs}: train {train_loss:.6g}, test {test_loss:.6g}",
            flush=True,
        )

    normalization = {
        "input_mean": input_mean.tolist(),
        "input_scale": input_scale.tolist(),
        "output_mean": output_mean.tolist(),
        "output_scale": output_scale.tolist(),
    }
    metadata = {
        "mode": mode,
        "dt": dt,
        "normalization": normalization,
        "input_features": ["thrust", "roll", "pitch", "yaw"] if mode == "z_psi" else ["u"],
        "output_features": ["az", "wdotz"] if mode == "z_psi" else ["az"],
        "hidden_size": hidden_size,
        "head_activation": "linear",
        "seed": seed,
        "inputs": str(Path(inputs).resolve()),
        "outputs": str(Path(outputs).resolve()),
        "input_format": input_format,
        "output_format": output_format,
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "rnn.pt"
    model.cpu()
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, checkpoint)
    (output_dir / "training.json").write_text(
        json.dumps({"metadata": metadata, "history": history}, indent=2) + "\n"
    )
    return model.eval(), checkpoint


def main(argv=None):
    """Train either dynamics model without changing the supplied checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["z", "z_psi"], default="z")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--input-format", choices=["controls", "motors"], default="controls")
    parser.add_argument("--output-format", choices=["selected", "plant"], default="selected")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Defaults to CUDA when available")
    args = parser.parse_args(argv)
    output_dir = args.output_dir or (
        PROJECT_ROOT / "results/model_training" / args.mode / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    torch.set_num_threads(1)
    _, checkpoint = train(
        args.inputs,
        args.outputs,
        output_dir,
        mode=args.mode,
        input_format=args.input_format,
        output_format=args.output_format,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_rows=args.max_trajectories,
        seed=args.seed,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(f"Saved model and training normalization: {checkpoint}")


if __name__ == "__main__":
    main()
