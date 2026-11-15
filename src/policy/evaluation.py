"""Policy evaluation and training plots."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np


def run_episode(env, policy, seed=None, max_steps=None):
    """Return states, actions, return, and outcome for a callable policy.

    The callable receives observations and returns environment actions.
    Recurrent memory belongs to the environment and is cleared at reset.
    """
    obs, info = env.reset(seed=seed)
    base = env.unwrapped
    states, actions, total_reward = [base.state.copy()], [], 0.0
    horizon = base.max_steps if max_steps is None else max_steps
    for _ in range(horizon):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(base.state.copy())
        actions.append(action)
        total_reward += reward
        if terminated or truncated:
            break
    return np.asarray(states), actions, total_reward, info["outcome"]


def evaluate(env, policy, n=50, seed=0):
    """Return mean reward and successful-mission fraction over n episodes."""
    returns, successes = [], 0
    for index in range(n):
        _, _, reward, outcome = run_episode(env, policy, seed=seed + index)
        returns.append(reward)
        successes += outcome in ("landed", ("landed", "has_turned"))
    returns = np.asarray(returns)
    print(
        f"mean return: {returns.mean():7.1f} ± {returns.std():5.1f} | "
        f"landed: {successes}/{n}"
    )
    return returns.mean(), successes / n


def evaluate_policy(agent, env, output_dir, *, seed=100, show=False):
    """Save an altitude mission's metrics, trajectory and flight plot.

    Commands are logged in motor units, including the DQN command grid.
    Only current observations reach the policy; references are for plotting.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = env.unwrapped
    obs, info = env.reset(seed=seed)
    states, actions, rewards = [base.state.copy()], [], []
    for _ in range(base.max_steps):
        action = agent.predict(obs, deterministic=True)[0]
        command = env.action(action) if isinstance(env, gym.ActionWrapper) else action
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(base.state.copy())
        actions.append(np.asarray(command, dtype=np.float32).reshape(1))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    states, actions = np.asarray(states), np.asarray(actions)
    t = np.arange(len(states)) * base.p.dt
    references = np.asarray([base.mission(float(ti)) for ti in t])
    commands = np.r_[actions[:, 0], np.nan]
    changes = np.diff(np.vstack([np.zeros((1, 1)), actions]), axis=0)
    metrics = {
        "outcome": info["outcome"],
        "landed": info["outcome"] == "landed",
        "return": float(np.sum(rewards)),
        "steps": len(actions),
        "duration_s": float(t[-1]),
        "peak_altitude_m": float(states[:, 0].max()),
        "final_altitude_m": float(states[-1, 0]),
        "final_vertical_velocity_m_s": float(states[-1, 1]),
        "altitude_mae_m": float(np.mean(np.abs(states[1:, 0] - references[1:]))),
        "rms_delta_u": float(np.sqrt(np.mean(changes ** 2))),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez(
        output_dir / "trajectory.npz", t=t, states=states, actions=actions,
        rewards=np.asarray(rewards), references=references,
    )
    np.savetxt(
        output_dir / "trajectory.csv",
        np.column_stack([t, states, references, commands]),
        delimiter=",", header="t,z,vz,az,z_ref,u", comments="",
    )

    reference_t = np.arange(base.max_steps + 1) * base.p.dt
    full_reference = [base.mission(float(ti)) for ti in reference_t]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    axes[0].plot(t, states[:, 0], label="Policy")
    axes[0].step(reference_t, full_reference, where="post", color="black",
                 linestyle="--", label="Mission target")
    axes[0].legend()
    axes[0].set_title(
        f"{type(agent).__name__}: {info['outcome']} at {t[-1]:.2f} s"
    )
    axes[1].plot(t, states[:, 1])
    axes[2].step(t, np.r_[actions[:, 0], actions[-1, 0]], where="post")
    labels = ["Altitude z (m)", "Vertical velocity (m/s)", "Motor command"]
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.axvline(t[-1], color="tab:blue", linestyle=":", alpha=0.4)
        axis.grid(alpha=0.3)
    axes[-1].set(xlabel="Time (s)", xlim=(0, reference_t[-1]))
    fig.tight_layout()
    fig.savefig(output_dir / "flight.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return metrics


def evaluate_rotation(agent, env, output_dir, *, seed=0, show=False):
    """Save the z/yaw mission with commands restored to physical units.

    Yaw stays unwrapped so a complete turn remains distinct from zero.
    Pitch and roll are held at zero by this environment and are not plotted.
    """
    import matplotlib.pyplot as plt

    states, actions, total, outcome = run_episode(
        env, lambda obs: agent.predict(obs, deterministic=True)[0], seed=seed,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = np.asarray(agent.action_scale, dtype=np.float32)
    offset = np.asarray(agent.action_offset, dtype=np.float32)
    actions = offset + scale * np.asarray(actions, dtype=np.float32)
    base = env.unwrapped
    t = np.arange(len(states)) * base.p.dt
    references = np.asarray([base.mission(float(ti)) for ti in t])
    # The final state has no following command.
    commands = np.vstack([actions, np.full((1, 2), np.nan)])
    trajectory = dict(
        t=t, z=states[:, 0], vz=states[:, 1], az=states[:, 2],
        psi=states[:, 11], psi_dot=states[:, 16], psi_ddot=states[:, 17],
        z_ref=references[:, 0], psi_ref=references[:, 1],
        thrust=commands[:, 0], yaw=commands[:, 1],
    )
    np.savez(output_dir / "trajectory.npz", **trajectory)
    np.savetxt(
        output_dir / "trajectory.csv", np.column_stack(list(trajectory.values())),
        delimiter=",", header=",".join(trajectory), comments="",
    )
    metrics = {
        "outcome": outcome,
        "success": outcome == ("landed", "has_turned"),
        "return": float(total),
        "steps": len(actions),
        "duration_s": float(t[-1]),
        "altitude_mae_m": float(np.mean(np.abs(states[1:, 0] - references[1:, 0]))),
        "yaw_mae_rad": float(np.mean(np.abs(states[1:, 11] - references[1:, 1]))),
        "peak_altitude_m": float(states[:, 0].max()),
        "altitude_hold_time_s": float(base.altitude_hold_time),
        "minimum_altitude_m": float(states[:, 0].min()),
        "final_altitude_m": float(states[-1, 0]),
        "final_vz_m_s": float(states[-1, 1]),
        "final_yaw_rad": float(states[-1, 11]),
        "final_yaw_speed_rad_s": float(states[-1, 16]),
    }

    reference_t = np.arange(base.max_steps + 1) * base.p.dt
    full_reference = np.asarray([base.mission(float(ti)) for ti in reference_t])
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    for axis, index, target, ylabel in (
        (axes[0], 0, 0, "Altitude z (m)"),
        (axes[1], 11, 1, "Yaw psi (rad)"),
    ):
        axis.plot(t, states[:, index], label="Policy")
        axis.plot(reference_t, full_reference[:, target], "k--", label="Mission")
        axis.set_ylabel(ylabel)
    for index, name in enumerate(("thrust", "yaw")):
        axes[2].step(t, np.r_[actions[:, index], actions[-1, index]], where="post", label=name)
    axes[0].set_title(f"PPO: {', '.join(outcome)} at {t[-1]:.2f} s")
    axes[2].set(xlabel="Time (s)", ylabel="Command", xlim=(0, reference_t[-1]))
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "flight.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return metrics


def training_step_offset(log_dir):
    """Return the steps already trained before this run's monitor logs."""
    for path in Path(log_dir).glob("*_config.json"):
        config = json.loads(path.read_text())
        warm_start = config.get("metadata", {}).get("warm_start", {})
        if warm_start:
            return int(warm_start["initial_timesteps"])
    return 0


def plot_learning_curve(log_dir, output_path, *, show=False):
    """Plot episode returns and deterministic evaluations."""
    import matplotlib.pyplot as plt
    from stable_baselines3.common.results_plotter import load_results

    log_dir = Path(log_dir)
    history = load_results(str(log_dir))
    returns = history["r"].to_numpy()
    steps = history["l"].cumsum().to_numpy() + training_step_offset(log_dir)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(steps, returns, alpha=0.25, label="Episode return")
    window = min(20, len(returns))
    if window:
        ax.plot(steps[window - 1:], np.convolve(returns, np.ones(window) / window, "valid"),
                label=f"Mean over {window} episodes")
    evaluation_file = log_dir / "evaluations.npz"
    if evaluation_file.is_file():
        with np.load(evaluation_file) as evaluations:
            ax.plot(evaluations["timesteps"], evaluations["results"].mean(axis=1),
                    marker=".", label="Deterministic evaluation")
    ax.set(xlabel="Environment steps", ylabel="Episode return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
