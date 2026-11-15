"""Portable dynamics exports with the normalization actually used by Python."""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.io import savemat


def export_onnx(model, path):
    """Export explicit recurrent state and dynamic sequence/batch dimensions."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rnn=model.rnn
    example=(torch.zeros(1,1,rnn.input_size),torch.zeros(rnn.num_layers,1,rnn.hidden_size))
    torch.onnx.export(model,example,str(path),input_names=['x','h'],output_names=['y','h_new'],
                      dynamic_axes={'x':{0:'batch',1:'time'},'h':{1:'batch'},
                                    'y':{0:'batch',1:'time'},'h_new':{1:'batch'}},
                      opset_version=17,dynamo=False)
    return path


def export_rnn_for_matlab(model, params, output_dir, mode):
    """Write a linear-head tanh RNN plus physical-unit golden sequences."""
    if not isinstance(model.head,nn.Linear):
        raise ValueError('The deployed altitude/yaw models require a linear RNN head')
    rotation=mode=='z_psi'
    mean=np.asarray(params.mean_u,dtype=np.float64).reshape(-1)
    scale=np.asarray(params.sigma_u,dtype=np.float64).reshape(-1)
    scale=np.where(scale==0,1,scale)
    output_mean=np.array([params.mean_az,params.mean_wz_dot] if rotation else [params.mean_az])
    output_scale=np.array([params.sigma_az,params.sigma_wz_dot] if rotation else [params.sigma_az])
    # Match the environment's float32 input normalization, including zero channels.
    raw=np.random.default_rng(0).uniform(-.2,.8,size=(64,model.rnn.input_size)).astype(np.float32)
    if rotation:raw[:,1:3]=0
    normalized=(raw-np.asarray(params.mean_u,dtype=np.float32))/scale.astype(np.float32)
    predictions=[]; hidden=None
    with torch.inference_mode():
        for sample in normalized:
            y,hidden=model(torch.from_numpy(sample).reshape(1,1,-1),hidden)
            predictions.append(y.numpy().reshape(-1)*output_scale+output_mean)
    data=dict(mode=mode,input_mean=mean,input_scale=scale,output_mean=output_mean,
              output_scale=output_scale,dt=float(params.dt),hidden_size=model.rnn.hidden_size,
              n_layers=model.rnn.num_layers,head_weight=model.head.weight.detach().numpy(),
              head_bias=model.head.bias.detach().numpy().reshape(-1,1),
              validation_inputs=raw,validation_outputs=np.asarray(predictions))
    state=model.state_dict()
    for layer in range(model.rnn.num_layers):
        for target,source in [('Wih','weight_ih'),('Whh','weight_hh'),('bih','bias_ih'),('bhh','bias_hh')]:
            value=state[f'rnn.{source}_l{layer}'].detach().numpy()
            data[f'{target}{layer+1}']=value.reshape(-1,1) if source.startswith('bias') else value
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/'rnn.mat';savemat(path,data)
    return path
