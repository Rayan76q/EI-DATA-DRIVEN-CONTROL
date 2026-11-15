"""Train the team's scalar altitude or four-input altitude/yaw dynamics RNN.

Run from the project root, e.g. python src/model/training/train.py --help.
The existing policy-compatible checkpoints are never overwritten implicitly.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
MOTOR_TO_CONTROL = np.array([
    [.25, .25, .25, .25], [0, .5, 0, -.5],
    [-.5, 0, .5, 0], [.25, -.25, .25, -.25],
], dtype=np.float32)


def read_csv(path, max_rows=None):
    """Read trajectories, tolerating only MATLAB's empty trailing column."""
    values = np.genfromtxt(path, delimiter=",", dtype=np.float32, ndmin=2, max_rows=max_rows)
    while values.shape[1] and np.isnan(values[:, -1]).all():
        values = values[:, :-1]
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Dataset must contain finite numeric trajectories: {path}")
    return values


def prepare_sequences(inputs, outputs, mode, input_format="controls", output_format="selected"):
    """Preserve trajectory boundaries and the team's [az, wdot_z] ordering."""
    input_width = 4 if input_format == "motors" or mode == "z_psi" else 1
    output_width = 18 if output_format == "plant" else 3 if output_format == "z_state" else 2 if mode == "z_psi" else 1
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
        if mode == "z":
            if not np.allclose(controls[..., 1:], 0, atol=1e-5):
                raise ValueError("Scalar z training requires equal motor commands; use z_psi for yaw data")
            x = controls[..., :1]
        else:
            x = controls
    if output_format == "plant":
        y = y[..., [8, 17] if mode == "z_psi" else [8]]
    elif output_format == "z_state":
        if mode != "z":
            raise ValueError("z_state outputs [z,vz,az] apply only to the altitude model")
        y = y[..., [2]]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Training sequences contain non-finite values")
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class ModelNN(nn.Module):
    """The team's tanh RNN with a linear acceleration output head."""

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        output, hidden = self.rnn(x, hidden)
        return self.head(output), hidden


def train(inputs, outputs, output_dir, *, mode="z", input_format="controls",
          output_format="selected", epochs=700, batch_size=32, seed=0,
          hidden_size=None, learning_rate=1e-3, max_rows=None, dt=.05, device=None):
    if mode not in {"z", "z_psi"} or epochs <= 0 or batch_size <= 0 or dt <= 0:
        raise ValueError("Choose a supported mode and positive training settings")
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(seed)
    x, y = prepare_sequences(read_csv(inputs, max_rows), read_csv(outputs, max_rows),
                             mode, input_format, output_format)
    if len(x) < 3:
        raise ValueError("At least three complete trajectories are required for train/test splitting")
    order = rng.permutation(len(x))
    split = min(len(x)-1, max(1, int(.9*len(x))))
    train_indices, test_indices = order[:split], order[split:]
    input_mean = x[train_indices].mean(axis=(0, 1))
    input_scale = x[train_indices].std(axis=(0, 1)) + 1e-5
    output_mean = y[train_indices].mean(axis=(0, 1))
    output_scale = y[train_indices].std(axis=(0, 1)) + 1e-5
    xn, yn = (x-input_mean)/input_scale, (y-output_mean)/output_scale
    dataset = TensorDataset(torch.from_numpy(xn[train_indices]), torch.from_numpy(yn[train_indices]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    hidden_size = hidden_size or (64 if mode == "z_psi" else 32)
    model = ModelNN(x.shape[-1], y.shape[-1], hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_updates = epochs*len(loader)
    warmup = max(1, int(.1*total_updates))
    def rate(step):
        if step < warmup:
            return .001 + .999*(step+1)/warmup
        return .5*(1+np.cos(np.pi*(step-warmup)/max(1,total_updates-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    history = []
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            prediction, _ = model(xb)
            loss = criterion(prediction, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            prediction, _ = model(torch.from_numpy(xn[test_indices]).to(device))
            test_loss = float(criterion(prediction, torch.from_numpy(yn[test_indices]).to(device)))
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),test_loss=test_loss))
        print(f"Epoch {epoch+1}/{epochs}: train {history[-1]['train_loss']:.6g}, test {test_loss:.6g}",flush=True)
    normalization = dict(input_mean=input_mean.tolist(),input_scale=input_scale.tolist(),
                         output_mean=output_mean.tolist(),output_scale=output_scale.tolist())
    metadata = dict(mode=mode,dt=dt,normalization=normalization,
                    input_features=["thrust","roll","pitch","yaw"] if mode=="z_psi" else ["u"],
                    output_features=["az","wdotz"] if mode=="z_psi" else ["az"],
                    hidden_size=hidden_size,head_activation="linear",seed=seed,
                    inputs=str(Path(inputs).resolve()),outputs=str(Path(outputs).resolve()),
                    input_format=input_format,output_format=output_format,
                    train_indices=train_indices.tolist(),test_indices=test_indices.tolist())
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=output_dir/'rnn.pt'
    model.cpu()
    torch.save(dict(state_dict=model.state_dict(),metadata=metadata),checkpoint)
    (output_dir/'training.json').write_text(json.dumps(dict(metadata=metadata,history=history),indent=2)+'\n')
    return model.eval(),checkpoint


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['z','z_psi'],default='z')
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--outputs',type=Path,required=True)
    parser.add_argument('--input-format',choices=['controls','motors'],default='controls')
    parser.add_argument('--output-format',choices=['selected','plant','z_state'],default='selected')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=700)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--max-trajectories',type=int)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--hidden-size',type=int)
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--device',choices=['cpu','cuda'],help='Defaults to CUDA when available')
    parser.add_argument('--onnx',action='store_true',help='Also export a dynamic-shape ONNX model')
    args=parser.parse_args(argv)
    output=args.output_dir or PROJECT_ROOT/'results/model_training'/args.mode/datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.set_num_threads(1)
    model,checkpoint=train(args.inputs,args.outputs,output,mode=args.mode,input_format=args.input_format,
                       output_format=args.output_format,epochs=args.epochs,batch_size=args.batch_size,
                       max_rows=args.max_trajectories,seed=args.seed,hidden_size=args.hidden_size,
                       learning_rate=args.learning_rate,device=args.device)
    if args.onnx:
        from model.export_rnn import export_onnx
        export_onnx(model,output/'rnn.onnx')
    print(f'Saved model and training normalization: {checkpoint}')


if __name__=='__main__':
    main()
