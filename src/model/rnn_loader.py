"""Load the altitude and rotation RNN checkpoints without running training code."""

import io
from pathlib import Path
import zipfile
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


class CheckpointRNN(nn.Module):
    """Reconstruct a tanh RNN and its output head from state-dict dimensions."""

    def __init__(self, weights, head_activation=None):
        super().__init__()
        hidden_size = weights["rnn.weight_hh_l0"].shape[1]
        input_size = weights["rnn.weight_ih_l0"].shape[1]
        num_layers = sum(key.startswith("rnn.weight_ih_l") for key in weights)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        if "head.weight" in weights:
            self.head = nn.Linear(hidden_size, weights["head.weight"].shape[0])
        else:
            # Activation types are not stored in a state dict: supply the one
            # used during training rather than guessing from tensor shapes.
            activations = {"tanh": nn.Tanh, "relu": nn.ReLU}
            if head_activation not in activations:
                raise ValueError("Specify the training head_activation: 'tanh' or 'relu'")
            head_size = weights["head.0.weight"].shape[0]
            output_size = weights["head.2.weight"].shape[0]
            self.head = nn.Sequential(
                nn.Linear(hidden_size, head_size),
                activations[head_activation](),
                nn.Linear(head_size, output_size),
            )

        self.load_state_dict(weights, strict=True)

    def forward(self, x, hidden=None):
        output, hidden = self.rnn(x, hidden)
        return self.head(output), hidden


def load_rnn_model(path, *, head_activation=None):
    """Load a PyTorch state dict, including an extracted checkpoint directory.

    The original single-output linear head and the rotation model's two-layer
    head are supported. All parameters must match; no weights are ignored.
    """
    path = Path(path)
    if path.is_dir():
        if not (path / "data.pkl").is_file():
            raise FileNotFoundError(f"Missing checkpoint data.pkl in {path}")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", strict_timestamps=False) as archive:
            for file in sorted(path.rglob("*")):
                if file.is_file():
                    archive.write(file, f"checkpoint/{file.relative_to(path).as_posix()}")
        buffer.seek(0)
        payload = torch.load(buffer, map_location="cpu", weights_only=True)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)

    metadata = payload.get("metadata", {}) if "state_dict" in payload else {}
    weights = payload["state_dict"] if "state_dict" in payload else payload
    head_activation = head_activation or metadata.get("head_activation")
    model = CheckpointRNN(weights, head_activation=head_activation)
    model.training_metadata = metadata
    return model.eval().requires_grad_(False)


def checkpoint_parameters(model, defaults, mode):
    """Use saved training statistics for new models; preserve legacy defaults."""
    metadata = getattr(model, "training_metadata", {})
    if not metadata:
        return defaults
    if metadata.get("mode") != mode:
        raise ValueError(f"This environment requires an RNN trained for {mode}")
    normal = metadata["normalization"]
    means = np.asarray(normal["input_mean"], dtype=np.float32)
    scales = np.asarray(normal["input_scale"], dtype=np.float32)
    out_mean = np.asarray(normal["output_mean"], dtype=np.float32)
    out_scale = np.asarray(normal["output_scale"], dtype=np.float32)
    expected = (4, 2) if mode == "z_psi" else (1, 1)
    if (means.shape != (expected[0],) or scales.shape != means.shape
            or out_mean.shape != (expected[1],) or out_scale.shape != out_mean.shape
            or not all(np.isfinite(a).all() for a in [means,scales,out_mean,out_scale])
            or np.any(scales <= 0) or np.any(out_scale <= 0)):
        raise ValueError("Invalid RNN training normalization")
    params = SimpleNamespace(**vars(defaults))
    params.dt = float(metadata["dt"])
    if mode == "z":
        params.mean_u, params.sigma_u = float(means[0]), float(scales[0])
    else:
        params.mean_u, params.sigma_u = means, scales
        params.mean_wz_dot, params.sigma_wz_dot = float(out_mean[1]),float(out_scale[1])
    params.mean_az, params.sigma_az = float(out_mean[0]),float(out_scale[0])
    return params
