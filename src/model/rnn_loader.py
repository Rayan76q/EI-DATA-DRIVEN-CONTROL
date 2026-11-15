"""Load the altitude and rotation RNN checkpoints without running training code."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


class CheckpointRNN(nn.Module):
    """Reconstruct a tanh RNN and its output head from state-dict dimensions."""

    def __init__(self, weights):
        super().__init__()
        hidden_size = weights["rnn.weight_hh_l0"].shape[1]
        input_size = weights["rnn.weight_ih_l0"].shape[1]
        num_layers = sum(key.startswith("rnn.weight_ih_l") for key in weights)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        self.head = nn.Linear(hidden_size, weights["head.weight"].shape[0])
        self.load_state_dict(weights, strict=True)

    def forward(self, x, hidden=None):
        output, hidden = self.rnn(x, hidden)
        return self.head(output), hidden


def load_rnn_model(path):
    """Load saved weights or a training bundle containing weights and statistics.

    The two supplied checkpoints store weights only. New training runs also
    save their normalization so the environment can use the same physical units.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    bundled = "state_dict" in payload
    weights = payload["state_dict"] if bundled else payload
    model = CheckpointRNN(weights)
    model.training_metadata = payload.get("metadata", {}) if bundled else {}
    return model.eval().requires_grad_(False)


def checkpoint_parameters(model, defaults, mode):
    """Apply bundle statistics, or use the supplied checkpoint's fixed defaults."""
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
    input_width, output_width = (4, 2) if mode == "z_psi" else (1, 1)
    valid_shapes = (
        means.shape == scales.shape == (input_width,)
        and out_mean.shape == out_scale.shape == (output_width,)
    )
    finite = all(np.isfinite(value).all() for value in (means, scales, out_mean, out_scale))
    if not valid_shapes or not finite or np.any(scales <= 0) or np.any(out_scale <= 0):
        raise ValueError("Invalid RNN training normalization")
    params = SimpleNamespace(**vars(defaults))
    params.dt = float(metadata["dt"])
    if mode == "z":
        params.mean_u, params.sigma_u = float(means[0]), float(scales[0])
    else:
        params.mean_u, params.sigma_u = means, scales
        params.mean_wz_dot, params.sigma_wz_dot = float(out_mean[1]), float(out_scale[1])
    params.mean_az, params.sigma_az = float(out_mean[0]), float(out_scale[0])
    return params
