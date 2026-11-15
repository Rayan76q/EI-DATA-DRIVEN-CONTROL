"""Export recurrent dynamics and validation samples for MATLAB."""

from pathlib import Path

import numpy as np
from scipy.io import savemat
import torch
from torch import nn


def export_rnn_for_matlab(model, params, output_dir, mode):
    """Save RNN weights, physical normalization and a reproducible test sequence.

    MATLAB checks its recurrence against the Python predictions before running
    the plant. Inputs and outputs in this validation sequence use physical units.
    """
    if not isinstance(model.head, nn.Linear):
        raise ValueError("The deployed dynamics RNN requires a linear output head")

    rotation = mode == "z_psi"
    input_mean = np.asarray(params.mean_u, dtype=np.float64).reshape(-1)
    input_scale = np.asarray(params.sigma_u, dtype=np.float64).reshape(-1)
    input_scale = np.where(input_scale == 0, 1, input_scale)
    output_mean = [params.mean_az]
    output_scale = [params.sigma_az]
    if rotation:
        output_mean.append(params.mean_wz_dot)
        output_scale.append(params.sigma_wz_dot)
    output_mean = np.asarray(output_mean)
    output_scale = np.asarray(output_scale)

    raw = np.random.default_rng(0).uniform(
        -0.2, 0.8, size=(64, model.rnn.input_size)
    ).astype(np.float32)
    if rotation:
        raw[:, 1:3] = 0

    # Match the environment's float32 normalization, including unused channels.
    normalized = (raw - input_mean.astype(np.float32)) / input_scale.astype(np.float32)
    predictions = []
    hidden = None
    with torch.inference_mode():
        for sample in normalized:
            prediction, hidden = model(torch.from_numpy(sample).reshape(1, 1, -1), hidden)
            predictions.append(prediction.numpy().reshape(-1) * output_scale + output_mean)

    data = {
        "mode": mode,
        "input_mean": input_mean,
        "input_scale": input_scale,
        "output_mean": output_mean,
        "output_scale": output_scale,
        "dt": float(params.dt),
        "hidden_size": model.rnn.hidden_size,
        "n_layers": model.rnn.num_layers,
        "head_weight": model.head.weight.detach().numpy(),
        "head_bias": model.head.bias.detach().numpy().reshape(-1, 1),
        "validation_inputs": raw,
        "validation_outputs": np.asarray(predictions),
    }
    state = model.state_dict()
    matlab_names = {
        "Wih": "weight_ih",
        "Whh": "weight_hh",
        "bih": "bias_ih",
        "bhh": "bias_hh",
    }
    for layer in range(model.rnn.num_layers):
        for target, source in matlab_names.items():
            value = state[f"rnn.{source}_l{layer}"].detach().numpy()
            if source.startswith("bias"):
                value = value.reshape(-1, 1)
            data[f"{target}{layer + 1}"] = value

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rnn.mat"
    savemat(path, data)
    return path
