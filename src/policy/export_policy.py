"""Export altitude and altitude/yaw policies for MATLAB/Simulink."""

from pathlib import Path

import numpy as np
import torch.nn as nn
from gymnasium import spaces
from scipy.io import savemat
from stable_baselines3.common.torch_layers import FlattenExtractor

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
    extractor = (agent.q_net.features_extractor if name == "DQN" else
                 agent.actor.features_extractor if name in {"SAC", "TD3"} else
                 agent.policy.pi_features_extractor)
    if hasattr(extractor, "scale"):
        observation_scale = extractor.scale.detach().cpu().double().numpy()
    elif isinstance(extractor, FlattenExtractor):
        observation_scale = np.ones(agent.observation_space.shape, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported policy feature extractor: {type(extractor).__name__}")
    if (observation_scale.shape != agent.observation_space.shape
            or not np.isfinite(observation_scale).all() or np.any(observation_scale <= 0)):
        raise ValueError("Policy observation scales must be finite and positive")
    P["observation_scale"] = observation_scale.reshape(-1, 1)
    if name != "DQN":
        scale = getattr(agent, "action_scale", None)
        offset = getattr(agent, "action_offset", None)
        P["action_scale"] = np.asarray(np.ones_like(action_low) if scale is None else scale, dtype=np.float64)
        P["action_offset"] = np.asarray(np.zeros_like(action_low) if offset is None else offset, dtype=np.float64)
        if (P["action_scale"].shape != action_low.shape or P["action_offset"].shape != action_low.shape
                or not np.isfinite(P["action_scale"]).all() or np.any(P["action_scale"] <= 0)
                or not np.isfinite(P["action_offset"]).all()):
            raise ValueError("Saved action scaling must match the policy action space")
    return P


def _predict_matrices(policy, obs):
    """Evaluate the matrices as rl_controller.m does."""
    x = np.asarray(obs, dtype=np.float64).reshape(-1, 1)
    x = x / np.asarray(policy.get("observation_scale", np.ones_like(x))).reshape(-1, 1)
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
    if "action_scale" in policy:
        x = policy["action_offset"].reshape(-1, 1) + policy["action_scale"].reshape(-1, 1) * x
    return x.reshape(-1)


def export_for_matlab(agent, env, output_dir="."):
    """Export a DQN, SAC, TD3 or PPO MLP policy; return both MAT-file paths.

    policy.mat includes the mission and sample settings. traj_model.mat
    contains the flight log. Commands are checked against SB3 first.
    """
    base = env.unwrapped
    is_dqn = type(agent).__name__ == "DQN"
    signature = (env.observation_space.shape, base.action_space.shape)
    if signature not in {((3,), (1,)), ((6,), (2,))}:
        raise ValueError("MATLAB export supports altitude (3/1) and altitude/yaw (6/2)")
    rotation = signature[0] == (6,)
    if agent.observation_space != env.observation_space:
        raise ValueError("Policy and export environment observations differ")
    if is_dqn:
        if not isinstance(env.action_space, spaces.Discrete) or env.action_space != agent.action_space:
            raise ValueError("DQN export requires the matching discrete action environment")
        commands = np.asarray(env.get_wrapper_attr("commands"), dtype=np.float64).reshape(-1)
    else:
        if env.action_space != agent.action_space:
            raise ValueError("Policy and export environment actions differ; apply the saved action wrapper")
        if getattr(agent, "action_scale", None) is not None:
            if not np.allclose(env.get_wrapper_attr("scale"), agent.action_scale):
                raise ValueError("Export environment uses different action scaling")
            if not np.allclose(env.get_wrapper_attr("offset"), agent.action_offset):
                raise ValueError("Export environment uses different action offsets")
        commands = None

    policy = _policy_matrices(agent, commands)
    obs, info = env.reset(seed=0)
    initial_az = float(obs[2])
    mission_t = np.arange(base.max_steps + 1, dtype=np.float64) * base.p.dt
    references = np.asarray([base.mission(float(t)) for t in mission_t], dtype=np.float64)
    mixer = (np.array([[1, 1], [1, -1], [1, 1], [1, -1]], dtype=np.float64)
             if rotation else np.ones((4, 1)))
    policy.update(
        schema_version=2,
        mode="z_psi" if rotation else "z",
        mission_t=mission_t.reshape(-1, 1),
        mission_z_ref=(references[:, 0] if rotation else references).reshape(-1, 1),
        Ts=float(base.p.dt),
        initial_az=initial_az,
        g=float(base.p.g),
        horizon=float(base.max_steps * base.p.dt),
        max_steps=int(base.max_steps),
        observation_order="z-z_ref,vz,az,psi-psi_ref,wz,wdotz" if rotation else "z-z_ref,vz,az",
        action_units="thrust,yaw" if rotation else "raw_motor_command",
        motor_mixer=mixer,
    )
    if rotation:
        policy.update(mission_psi_ref=references[:, 1].reshape(-1, 1),
                      initial_wdotz=float(base.state[17]))
    measurements = (*MEASUREMENTS, "psi", "wz", "wdotz_prev", "psi_ref", "yaw_prev") if rotation else MEASUREMENTS
    log = {k: [] for k in (*measurements, "obs", "u", "motors")}
    max_error = 0.0
    previous = np.zeros(base.action_space.shape)
    for _ in range(base.max_steps):
        action = agent.predict(obs, deterministic=True)[0]
        if is_dqn:
            action = int(np.asarray(action).item())
            u = np.array([commands[action]])
        else:
            u = policy["action_offset"] + policy["action_scale"] * np.asarray(action).reshape(-1)
        exported_u = _predict_matrices(policy, obs)
        max_error = max(max_error, float(np.max(np.abs(u - exported_u))))
        if not np.allclose(u, exported_u, atol=1e-4, rtol=1e-5):
            raise ValueError(f"Policy output mismatch: export={exported_u}, SB3={u}")
        measured = dict(t=base.t, z=base.state[0], vz=base.state[1], az_prev=obs[2],
                        z_ref=references[base.steps, 0] if rotation else references[base.steps],
                        u_prev=previous[0])
        if rotation:
            measured.update(psi=base.state[11], wz=base.state[16], wdotz_prev=base.state[17],
                            psi_ref=references[base.steps, 1], yaw_prev=previous[1])
        for k in measurements:
            log[k].append(float(measured[k]))
        log["obs"].append(np.asarray(obs, dtype=np.float64).reshape(-1))
        log["u"].append(u)
        log["motors"].append(mixer @ u)
        previous = u.copy()
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    traj = {k: np.array(v) for k, v in log.items()}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "policy.mat"
    trajectory_path = output_dir / "traj_model.mat"
    # Independent channel probes exercise scaling and clipping beyond this flight.
    dimension = agent.observation_space.shape[0]
    probes = np.vstack([np.zeros((1, dimension)), np.eye(dimension), -np.eye(dimension),
                        10 * np.eye(dimension), -10 * np.eye(dimension), traj["obs"]])
    expected = []
    for observation in probes:
        action = agent.predict(observation.astype(np.float32), deterministic=True)[0]
        raw = (np.array([commands[int(action)]]) if is_dqn else
               policy["action_offset"] + policy["action_scale"] * np.asarray(action).reshape(-1))
        if not np.allclose(_predict_matrices(policy, observation), raw, atol=1e-4, rtol=1e-5):
            raise ValueError("Exported actor failed independent observation probes")
        expected.append(raw)
    policy["validation_obs"] = probes
    policy["validation_actions"] = np.asarray(expected)
    savemat(policy_path, policy)
    savemat(trajectory_path, {**{k: traj[k].reshape(-1, 1) for k in measurements},
                             "u": traj["u"], "obs": traj["obs"], "motors": traj["motors"]})
    if base.rnn_model is not None:
        from model.export_rnn import export_rnn_for_matlab
        export_rnn_for_matlab(base.rnn_model, base.p, output_dir, policy["mode"])
    print(f"Exported {policy_path} and {trajectory_path} ({len(traj['t'])} steps; "
          f"command error {max_error:.2g})")
    return policy_path, trajectory_path
