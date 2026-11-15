"""Evaluate altitude or altitude/yaw RNN dynamics in physical units."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from model.rnn_loader import load_rnn_model, checkpoint_parameters
from model.training.train import read_csv

PROJECT_ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['z','z_psi'],default='z')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--input-csv',type=Path,help='One flattened trajectory per row; physical controls')
    parser.add_argument('--target-csv',type=Path,help='Selected physical accelerations in the first row')
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--command',type=float,default=.394024,help='Constant collective command')
    parser.add_argument('--yaw-command',type=float,default=0.)
    parser.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'results/model')
    args=parser.parse_args()
    if args.steps<1 or not np.isfinite([args.command,args.yaw_command]).all():
        parser.error('Use a positive step count and finite commands')
    rotation=args.mode=='z_psi'
    if rotation:
        from policy.lib.droneEnvRotation import PARAMS
        default_checkpoint='rnn_model_thrust_speeds.pt'
    else:
        from policy.lib.droneEnv import PARAMS
        default_checkpoint='rnn_z.pt'
    checkpoint=args.checkpoint or PROJECT_ROOT/'src/model/checkpoints'/default_checkpoint
    model=load_rnn_model(checkpoint)
    params=checkpoint_parameters(model,PARAMS,args.mode)
    inputs,outputs=(4,2) if rotation else (1,1)
    if (model.rnn.input_size!=inputs or not isinstance(model.head,nn.Linear)
            or model.head.out_features!=outputs):
        parser.error('The checkpoint architecture does not match the selected mode')
    commands=np.zeros((args.steps,inputs),dtype=np.float32)
    commands[:,0]=args.command
    if rotation:commands[:,3]=args.yaw_command
    if args.input_csv:
        row=read_csv(args.input_csv,max_rows=1)[0]
        if len(row)%inputs:
            parser.error(f'Input CSV requires {inputs} controls per time step')
        commands=row.reshape(-1,inputs)[:args.steps]
    targets=None
    if args.target_csv:
        row=read_csv(args.target_csv,max_rows=1)[0]
        if len(row)%outputs or len(row)<len(commands)*outputs:
            parser.error('Target CSV must have matching selected acceleration samples')
        targets=row.reshape(-1,outputs)[:len(commands)]
    mean=np.asarray(params.mean_u,dtype=np.float32)
    scale=np.where(np.asarray(params.sigma_u)==0,1,params.sigma_u).astype(np.float32)
    normalized=(commands-mean)/scale
    output_mean=np.array([params.mean_az,params.mean_wz_dot] if rotation else [params.mean_az])
    output_scale=np.array([params.sigma_az,params.sigma_wz_dot] if rotation else [params.sigma_az])
    predictions=[];hidden=None
    with torch.inference_mode():
        for command in normalized:
            output,hidden=model(torch.from_numpy(command).reshape(1,1,inputs),hidden)
            predictions.append(output.numpy().reshape(-1)*output_scale+output_mean)
    predictions=np.asarray(predictions)
    times=(np.arange(len(commands))+1)*params.dt
    names=['az','wdotz'] if rotation else ['az']
    input_names=['thrust','roll','pitch','yaw'] if rotation else ['u']
    columns=[times,*commands.T,*normalized.T,*predictions.T]
    header=['time_s',*[f'{n}_physical' for n in input_names],
            *[f'{n}_standardized' for n in input_names],*[f'{n}_predicted' for n in names]]
    metadata=dict(checkpoint=str(checkpoint.resolve()),mode=args.mode,samples=len(commands),
                  dt_s=params.dt,input_size=inputs,output_size=outputs,
                  hidden_size=model.rnn.hidden_size,input_mean=mean.tolist(),input_scale=scale.tolist(),
                  output_mean=output_mean.tolist(),output_scale=output_scale.tolist(),
                  final_prediction=predictions[-1].tolist())
    if targets is not None:
        error=predictions-targets
        columns.extend([*targets.T,*error.T])
        header.extend([*[f'{n}_target' for n in names],*[f'{n}_error' for n in names]])
        metadata.update(mae=np.abs(error).mean(axis=0).tolist(),rmse=np.sqrt((error**2).mean(axis=0)).tolist())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    np.savetxt(args.output_dir/'rnn_inference.csv',np.column_stack(columns),delimiter=',',
               header=','.join(header),comments='')
    fig,axes=plt.subplots(1+outputs,1,sharex=True,figsize=(10,3*(1+outputs)))
    for k in ([0,3] if rotation else [0]):
        axes[0].step(times,commands[:,k],where='post',label=input_names[k])
    axes[0].set_ylabel('Physical control')
    for k,name in enumerate(names):
        axes[k+1].plot(times,predictions[:,k],label='RNN')
        if targets is not None:axes[k+1].plot(times,targets[:,k],'--',label='Measured')
        axes[k+1].set_ylabel('az [m/s²]' if k==0 else 'wdotz [rad/s²]')
    for axis in axes:axis.grid(alpha=.3);axis.legend()
    axes[-1].set_xlabel('Time [s]');fig.tight_layout()
    fig.savefig(args.output_dir/'rnn_inference.png',dpi=150);plt.close(fig)
    (args.output_dir/'metrics.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Predicted {len(commands)} samples with {checkpoint}; results: {args.output_dir.resolve()}')


if __name__=='__main__':
    main()
