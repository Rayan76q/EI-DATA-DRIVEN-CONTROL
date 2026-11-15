"""Isolated PPO audit experiment: raw RNN, frozen dynamics, explicit reward."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from model.rnn_loader import load_rnn_model

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=Path)
args = parser.parse_args()
torch.set_num_threads(1)
env_source = ROOT / "src/policy/lib/droneEnvRotation.py"
frozen_env = OUT / "frozen_droneEnvRotation.py"
if not frozen_env.exists():
    shutil.copyfile(env_source, frozen_env)
spec = importlib.util.spec_from_file_location("frozen_rotation", frozen_env)
rotation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rotation)
rnn_source = ROOT / "src/model/checkpoints/rnn_model_thrust_speeds.pt"
frozen_rnn = OUT / "rnn_model_thrust_speeds.pt"
if not frozen_rnn.exists():
    shutil.copyfile(rnn_source, frozen_rnn)
rnn = load_rnn_model(frozen_rnn, head_activation="tanh")

def huber(x):
    return .5*x*x if abs(x) <= 1 else abs(x)-.5

def trial_reward(old_state, state, u, outcome, target_ref=(0, 0), u_prev=None, dt=.05):
    if outcome[0] != "flying":
        return 100. if outcome == ("landed", "has_turned") else -25.
    ze, pe = float(state[0]-target_ref[0]), float(state[11]-target_ref[1])
    smooth = 0. if u_prev is None else float(np.abs(u-u_prev).sum())
    return dt * (2 + 4/(1+ze*ze) + 4/(1+pe*pe)
                 - huber(ze) - huber(pe) - .2*huber(state[1])
                 - .1*huber(state[16]) - .01*float(u[1])**2 - .02*smooth)

rotation.compute_reward = trial_reward

def mission(t):
    if t < 2: return 0., 0.
    if t < 5: return 2., 0.
    if t < 8: return 2., float(2*np.pi*(t-5)/3)
    return -.3, float(2*np.pi)

def make_env():
    return rotation.DroneEnv(rnn_model=rnn, mission=mission)

def evaluate(agent, tag):
    env = make_env()
    states, actions, total, outcome = rotation.run_episode(
        env, lambda obs: agent.predict(obs, deterministic=True)[0])
    refs = np.asarray([mission(i*.05) for i in range(1,len(states))])
    m = dict(step=int(agent.num_timesteps), tag=tag, outcome=list(outcome),
             success=outcome == ("landed", "has_turned"), return_=float(total),
             duration=(len(states)-1)*.05, z_mae=float(np.mean(abs(states[1:,0]-refs[:,0]))),
             psi_mae=float(np.mean(abs(states[1:,11]-refs[:,1]))),
             final_z=float(states[-1,0]), final_psi=float(states[-1,11]),
             final_vz=float(states[-1,1]), final_wz=float(states[-1,16]),
             peak_z=float(states[:,0].max()), min_z=float(states[:,0].min()))
    np.savez(OUT / f"trajectory_{tag}.npz", states=states, actions=actions)
    (OUT / f"metrics_{tag}.json").write_text(json.dumps(m,indent=2)+"\n")
    env.close()
    return m

class AuditCallback(BaseCallback):
    def __init__(self):
        super().__init__(); self.next_eval=20_000; self.best=(-1., -float("inf")); self.started=time.monotonic()
    def _on_training_start(self):
        self.next_eval = self.model.num_timesteps + 20_000
        self.record("initial")
    def record(self, tag):
        m=evaluate(self.model,tag)
        with (OUT / "evaluations.jsonl").open("a") as f: f.write(json.dumps(m)+"\n")
        rank=(float(m["success"]),m["return_"])
        if rank > self.best:
            self.best=rank; self.model.save(OUT / "best_model")
        print(json.dumps(dict(**m, elapsed_s=round(time.monotonic()-self.started,1))),flush=True)
    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.record(str(self.num_timesteps)); self.next_eval+=20_000
        return True

config=dict(seed=args.seed,n_envs=8,learning_rate=.0003,n_steps=256,batch_size=256,
            gamma=.995,gae_lambda=.95,n_epochs=10,ent_coef=.005,vf_coef=.5,
            policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),log_std_init=-1))
snapshot=dict(config=config,initial_actor_bias=[.8,0],initial_log_std=[-1,-4],
              mission="hold0 until2,climb2 until5,turn2pi until8,land-.3 until10",
              normalization={k:np.asarray(getattr(rotation,k)).tolist() for k in
              ["MEAN_U","SIGMA_U","MEAN_AZ","SIGMA_AZ","MEAN_WZ_DOT","SIGMA_WZ_DOT"]},
              rnn_sha256=hashlib.sha256(frozen_rnn.read_bytes()).hexdigest(),
              frozen_env_sha256=hashlib.sha256(frozen_env.read_bytes()).hexdigest(),
              torch=torch.__version__,reward="See trial_reward in this script")
(OUT / "config.json").write_text(json.dumps(snapshot,indent=2)+"\n")
env=make_vec_env(make_env,n_envs=config.pop("n_envs"),seed=args.seed)
try:
    if args.resume:
        agent=PPO.load(args.resume,env=env,device="cpu")
    else:
        agent=PPO("MlpPolicy",env,device="cpu",verbose=0,**config)
        with torch.no_grad():
            agent.policy.action_net.weight.zero_()
            agent.policy.action_net.bias.copy_(torch.tensor([.8,0.]))
            agent.policy.log_std.copy_(torch.tensor([-1.,-4.]))
    agent.set_logger(configure(str(OUT / "logs"),["csv"]))
    cb=AuditCallback()
    agent.learn(total_timesteps=args.steps,callback=cb,reset_num_timesteps=args.resume is None)
    agent.save(OUT / "final_model")
    cb.record("final")
    best=PPO.load(OUT / "best_model",device="cpu")
    cb.model=best
    print("BEST",json.dumps(evaluate(best,"best")),flush=True)
finally:
    env.close()
