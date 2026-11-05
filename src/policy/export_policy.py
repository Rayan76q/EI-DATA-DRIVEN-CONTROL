"""Export altitude policies and trajectories for MATLAB/Simulink."""

from pathlib import Path

import numpy as np
import torch.nn as nn
from gymnasium import spaces
from scipy.io import savemat

MEASUREMENTS = ("t", "z", "vz", "az_prev", "z_ref", "u_prev")


def _layers(modules):
    layers = []
    for mod in modules:
        if isinstance(mod, nn.Linear):
            layers.append([mod.weight.detach().cpu().double().numpy(),
                           mod.bias.detach().cpu().double().numpy(), "none"])
        elif isinstance(mod, (nn.ReLU, nn.Tanh)):
            layers[-1][2] = "relu" if isinstance(mod, nn.ReLU) else "tanh"
        elif not isinstance(mod, (nn.Flatten, nn.Identity)):
            raise ValueError(f"Unsupported export layer: {mod}")
    return layers


def _policy_matrices(agent, action_values=None):
    name = type(agent).__name__
    mean_clip = None
    if name == "DQN":
        if action_values is None:
            raise ValueError("DQN export requires the physical command grid")
        action_values = np.asarray(action_values, dtype=np.float64).reshape(-1)
        if len(action_values) != agent.action_space.n or not np.isfinite(action_values).all():
            raise ValueError("DQN command grid must contain one finite command per action")
        layers, output = _layers(agent.q_net.q_net), "discrete"
        action_low = np.array([action_values.min()])
        action_high = np.array([action_values.max()])
    elif name == "SAC":
        mu = list(agent.actor.mu) if isinstance(agent.actor.mu, nn.Sequential) else [agent.actor.mu]
        if isinstance(mu[-1], nn.Hardtanh):
            clip = mu.pop()
            mean_clip = (clip.min_val, clip.max_val)
        layers, output = _layers(agent.actor.latent_pi) + _layers(mu), "tanh_scale"
    elif name == "TD3":
        layers, output = _layers(agent.actor.mu), "scale"
    elif name == "PPO":
        layers = _layers(agent.policy.mlp_extractor.policy_net) + _layers([agent.policy.action_net])
        output = "clip"
    else:
        raise ValueError(f"Unsupported policy: {name}; expected DQN, SAC, TD3 or PPO")
    if name != "DQN":
        action_low = agent.action_space.low.astype(np.float64)
        action_high = agent.action_space.high.astype(np.float64)
    P = {"n_layers": len(layers), "activations": ",".join(a for _, _, a in layers), "output": output,
         "action_low": action_low, "action_high": action_high}
    if name == "DQN":
        P["action_values"] = action_values
    if mean_clip is not None:
        P["pre_output_low"], P["pre_output_high"] = mean_clip
    for i, (W, b, _) in enumerate(layers, start=1):
        P[f"W{i}"], P[f"b{i}"] = W, b.reshape(-1, 1)
    return P


def _predict_matrices(policy, obs):
    """Evaluate the matrices as rl_controller.m does."""
    x = np.asarray(obs, dtype=np.float64).reshape(-1, 1)
    activations = policy["activations"].split(",")
    for i, activation in enumerate(activations, start=1):
        x = policy[f"W{i}"] @ x + policy[f"b{i}"]
        if activation == "relu":
            x = np.maximum(x, 0)
        elif activation == "tanh":
            x = np.tanh(x)
    if "pre_output_low" in policy:
        x = np.clip(x, policy["pre_output_low"], policy["pre_output_high"])
    lo = policy["action_low"].reshape(-1, 1)
    hi = policy["action_high"].reshape(-1, 1)
    if policy["output"] == "discrete":
        x = np.array([policy["action_values"][np.argmax(x)]])
    elif policy["output"] == "tanh_scale":
        x = lo + 0.5 * (np.tanh(x) + 1) * (hi - lo)
    elif policy["output"] == "scale":
        x = lo + 0.5 * (x + 1) * (hi - lo)
    else:
        x = np.clip(x, lo, hi)
    return x.reshape(-1)


def export_for_matlab(agent, env, output_dir="."):
    """Export a DQN, SAC, TD3 or PPO MLP policy; return both MAT-file paths.

    policy.mat includes the mission and sample settings. traj_model.mat
    contains the flight log. Commands are checked against SB3 first.
    """
    base = env.unwrapped
    is_dqn = type(agent).__name__ == "DQN"
    if env.observation_space.shape != (3,) or base.action_space.shape != (1,):
        raise ValueError("MATLAB export requires 3 altitude observations and 1 action")
    if agent.observation_space.shape != (3,):
        raise ValueError("Policy must use 3 altitude observations and 1 action")
    if is_dqn:
        if not isinstance(env.action_space, spaces.Discrete) or env.action_space != agent.action_space:
            raise ValueError("DQN export requires the matching discrete action environment")
        commands = np.asarray(env.get_wrapper_attr("commands"), dtype=np.float64).reshape(-1)
    else:
        if env.action_space.shape != (1,) or agent.action_space.shape != (1,):
            raise ValueError("Policy must use 3 altitude observations and 1 action")
        commands = None

    policy = _policy_matrices(agent, commands)
    obs, info = env.reset(seed=0)
    initial_az = float(info["m"]["az_prev"])
    mission_t = np.arange(base.max_steps + 1, dtype=np.float64) * base.p.dt
    policy.update(
        mission_t=mission_t.reshape(-1, 1),
        mission_z_ref=np.asarray([base.mission(float(t)) for t in mission_t],
                                 dtype=np.float64).reshape(-1, 1),
        Ts=float(base.p.dt),
        initial_az=initial_az,
        g=float(base.p.g),
        horizon=float(base.max_steps * base.p.dt),
        max_steps=int(base.max_steps),
        observation_order="z-z_ref,vz,az",
        action_units="raw_motor_command",
    )
    log = {k: [] for k in (*MEASUREMENTS, "obs", "u")}
    max_error = 0.0
    for _ in range(base.max_steps):
        action = agent.predict(obs, deterministic=True)[0]
        if is_dqn:
            action = int(np.asarray(action).item())
            u = float(commands[action])
        else:
            u = float(np.asarray(action).reshape(-1)[0])
            action = np.array([u], dtype=np.float32)
        exported_u = float(_predict_matrices(policy, obs)[0])
        max_error = max(max_error, abs(u - exported_u))
        if not np.isclose(u, exported_u, atol=1e-4, rtol=1e-5):
            raise ValueError(f"Policy output mismatch: export={exported_u}, SB3={u}")
        for k in MEASUREMENTS:
            log[k].append(float(info["m"][k]))
        log["obs"].append(np.asarray(obs, dtype=np.float64).reshape(-1))
        log["u"].append(u)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    traj = {k: np.array(v) for k, v in log.items()}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "policy.mat"
    trajectory_path = output_dir / "traj_model.mat"
    savemat(policy_path, policy)
    savemat(trajectory_path, {**{k: traj[k].reshape(-1, 1) for k in (*MEASUREMENTS, "u")},
                             "obs": traj["obs"]})
    print(f"Exported {policy_path} and {trajectory_path} ({len(traj['t'])} steps; "
          f"command error {max_error:.2g})")
    return policy_path, trajectory_path
