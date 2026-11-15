"""Compare rotation reward weights against the unchanged learned dynamics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from model.rnn_loader import load_rnn_model
from policy.lib.droneEnvRotation import DroneEnv

CANDIDATES = {
    'damped': dict(alive=2, z_bonus=4, psi_bonus=4, z_cost=.5, psi_cost=.5,
                   vz_cost=.4, wz_cost=.2, yaw_cost=.01, smooth_cost=.03,
                   success=100, failure=25, scale_dt=True, bounded=False),
    'minimal': dict(alive=0, z_bonus=5, psi_bonus=5, z_cost=1, psi_cost=1,
                    vz_cost=.05, wz_cost=.02, yaw_cost=.01, smooth_cost=.05,
                    success=1000, failure=1000, scale_dt=False, bounded=False),
    'baseline': dict(alive=0, z_bonus=2, psi_bonus=2, z_cost=20, psi_cost=20,
                     vz_cost=1, wz_cost=.5, yaw_cost=1, smooth_cost=1,
                     success=1000, failure=1000, scale_dt=False, bounded=False),
    'balanced': dict(alive=1, z_bonus=4, psi_bonus=4, z_cost=1, psi_cost=.5,
                     vz_cost=.25, wz_cost=.03, yaw_cost=.005, smooth_cost=.01,
                     success=100, failure=50, scale_dt=True, bounded=False),
    'gentle': dict(alive=2, z_bonus=4, psi_bonus=4, z_cost=.5, psi_cost=.2,
                   vz_cost=.1, wz_cost=.02, yaw_cost=.001, smooth_cost=.005,
                   success=50, failure=20, scale_dt=True, bounded=False),
    'bounded': dict(alive=1, z_bonus=4, psi_bonus=4, z_cost=0, psi_cost=0,
                    vz_cost=.1, wz_cost=.05, yaw_cost=.01, smooth_cost=.01,
                    success=100, failure=25, scale_dt=True, bounded=True),
}

def mission(t):
    if t < 2: return (0., 0.)
    if t < 5: return (2., 0.)
    if t < 8: return (2., 2*np.pi*(t-5)/3)
    return (-.3, 2*np.pi)

def huber(x):
    return .5*x*x if abs(x) <= 1 else abs(x)-.5

def reward_for(state, action, outcome, target, previous, dt, p):
    if outcome[0] == 'landed':
        return p['success'] if outcome[1] == 'has_turned' else -p['failure']
    if outcome[0] in ('crashed','lost','timeout'): return -p['failure']
    ez, ep = float(state[0]-target[0]), float(state[11]-target[1])
    vz, wz = float(state[1]), float(state[16])
    velocity = (lambda x: np.tanh(x*x)) if p['bounded'] else huber
    reward = (p['alive'] + p['z_bonus']/(1+ez*ez) + p['psi_bonus']/(1+ep*ep)
              - p['z_cost']*huber(ez) - p['psi_cost']*huber(ep)
              - p['vz_cost']*velocity(vz) - p['wz_cost']*velocity(wz)
              - p['yaw_cost']*abs(float(action[1]))
              - p['smooth_cost']*float(np.abs(action-previous).sum()))
    return float(reward * dt if p['scale_dt'] else reward)

def metrics(agent, env):
    obs, info = env.reset(seed=100)
    states, actions, total = [env.state.copy()], [], 0.
    for _ in range(env.max_steps):
        action = agent.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(env.state.copy()); actions.append(action); total += reward
        if terminated or truncated: break
    states = np.asarray(states)
    t = np.arange(len(states))*env.p.dt
    refs = np.asarray([mission(ti) for ti in t])
    # Missing phases receive no completion credit, regardless of episode length.
    climb = (t >= 3) & (t < 8)
    turn = (t >= 5) & (t < 8)
    climb_fraction = float(np.sum(climb & (np.abs(states[:,0]-refs[:,0]) <= .5)) / 100)
    turn_fraction = float(np.sum(turn & (np.abs(states[:,11]-refs[:,1]) <= .5)) / 60)
    result = dict(steps=len(actions), duration_s=float(t[-1]), outcome=info['outcome'],
                  success=info['outcome']==('landed','has_turned'), episode_return=float(total),
                  altitude_mae_m=float(np.abs(states[1:,0]-refs[1:,0]).mean()),
                  yaw_mae_rad=float(np.abs(states[1:,11]-refs[1:,1]).mean()),
                  peak_z=float(states[:,0].max()), final_z=float(states[-1,0]),
                  final_psi=float(states[-1,11]), final_vz=float(states[-1,1]),
                  final_wz=float(states[-1,16]), climb_fraction=climb_fraction,
                  turn_fraction=turn_fraction)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('candidate', choices=CANDIDATES)
    parser.add_argument('--steps', type=int, default=150000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    output=Path(__file__).resolve().parent/f'{args.candidate}_seed{args.seed}'
    output.mkdir(exist_ok=True)
    p=CANDIDATES[args.candidate]
    (output/'reward_params.json').write_text(json.dumps(p,indent=2)+'\n')
    rnn=load_rnn_model(ROOT/'src/model/checkpoints/rnn_model_thrust_speeds.pt')
    class CandidateEnv(DroneEnv):
        def step(self, action):
            previous=self.u_prev.copy()
            obs, _, terminated, truncated, info=super().step(action)
            reward=reward_for(self.state,self.u_prev,info['outcome'],self.mission(self.t),previous,self.p.dt,p)
            return obs,reward,terminated,truncated,info
    def make(): return CandidateEnv(rnn_model=rnn,mission=mission)
    evaluation=make()
    class LogEvaluation(BaseCallback):
        def __init__(self):
            super().__init__(); self.best_return=-np.inf; self.history=[]
            if args.resume and (output/'evaluations.jsonl').is_file():
                self.best_return=max(json.loads(line)['episode_return'] for line in (output/'evaluations.jsonl').read_text().splitlines())
        def _on_step(self):
            if self.n_calls % 2000: return True
            row=dict(training_steps=self.num_timesteps, **metrics(self.model,evaluation))
            self.history.append(row)
            with (output/'evaluations.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            if row['episode_return'] > self.best_return:
                self.best_return=row['episode_return']; self.model.save(output/'best_model.zip')
            print(args.candidate,args.seed,json.dumps(row),flush=True)
            return True
    config=json.loads((ROOT/'src/policy/config/ppo.json').read_text())
    vector=make_vec_env(make,n_envs=config['n_envs'],seed=args.seed,monitor_dir=str(output))
    if args.resume:
        agent=PPO.load(args.resume,env=vector,device='cpu')
    else:
        agent=PPO('MlpPolicy',vector,seed=args.seed,device='cpu',verbose=0,**config['ppo'])
    callback=LogEvaluation()
    agent.learn(total_timesteps=args.steps,callback=callback,reset_num_timesteps=not bool(args.resume))
    agent.save(output/'final_model.zip')
    final=dict(training_steps=agent.num_timesteps,**metrics(agent,evaluation))
    (output/'final_metrics.json').write_text(json.dumps(final,indent=2)+'\n')
    print('FINAL',args.candidate,args.seed,json.dumps(final),flush=True)
    vector.close(); evaluation.close()

if __name__=='__main__': main()
