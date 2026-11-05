"""Policy evaluation and training plots."""

import json
from pathlib import Path

import numpy as np

from .lib.discrete_actions import DiscreteDroneActions


def evaluate_policy(agent, env, output_dir, *, seed=100, show=False):
    """Run a mission and save its trajectory, metrics and plot.

    Mission references are used for logging and plots; the policy receives
    only the current observation.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = env.unwrapped
    obs, info = env.reset(seed=seed)
    states, actions, rewards = [base.state.copy()], [], []
    for _ in range(base.max_steps):
        action = agent.predict(obs, deterministic=True)[0]
        command = env.action(action) if isinstance(env, DiscreteDroneActions) else action
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(base.state.copy())
        actions.append(np.asarray(command, dtype=np.float32).reshape(-1))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    states = np.asarray(states)
    actions = np.asarray(actions)
    t = np.arange(len(states)) * base.p.dt
    references = np.asarray([base.mission(float(ti)) for ti in t])
    z_ref = references if references.ndim == 1 else references[:, 0]
    outcome = info["outcome"]
    z_outcome = outcome[0] if isinstance(outcome, tuple) else outcome
    metrics = {
        "outcome": outcome,
        "landed": z_outcome == "landed",
        "return": float(np.sum(rewards)),
        "steps": len(actions),
        "duration_s": float(t[-1]),
        "peak_altitude_m": float(states[:, 0].max()),
        "final_altitude_m": float(states[-1, 0]),
        "final_vertical_velocity_m_s": float(states[-1, 1]),
        "altitude_mae_m": float(np.mean(np.abs(states[1:, 0] - z_ref[1:]))),
        "rms_delta_u": float(np.sqrt(np.mean(np.diff(
            np.vstack([np.zeros((1, actions.shape[1])), actions]), axis=0,
        ) ** 2))),
    }
    has_angles = states.shape[1] > 11
    if has_angles:
        metrics.update(
            final_pitch_rad=float(states[-1, 9]),
            final_roll_rad=float(states[-1, 10]),
            final_yaw_rad=float(states[-1, 11]),
            final_angular_speed_rad_s=float(np.linalg.norm(states[-1, [12, 14, 16]])),
        )

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez(
        output_dir / "trajectory.npz",
        t=t,
        states=states,
        actions=actions,
        rewards=np.asarray(rewards),
        references=references,
    )
    command_columns = np.vstack([actions, np.full((1, actions.shape[1]), np.nan)])
    command_names = ["u_base", "u_diff"] if actions.shape[1] == 2 else [
        "u" if actions.shape[1] == 1 else f"u{i + 1}" for i in range(actions.shape[1])
    ]
    state_names = ["z", "vz", "az"] if not has_angles else [
        "z", "vz", "az", "y", "vy", "ay", "x", "vx", "ax",
        "theta", "phi", "psi", "wx", "wx_dot", "wy", "wy_dot", "wz", "wz_dot",
    ]
    np.savetxt(
        output_dir / "trajectory.csv",
        np.column_stack([t, states, z_ref, command_columns]),
        delimiter=",",
        header=",".join(["t", *state_names, "z_ref", *command_names]),
        comments="",
    )

    fig, axes = plt.subplots(6 if has_angles else 3, 1, sharex=True,
                             figsize=(9, 12 if has_angles else 7))
    axes[0].plot(t, states[:, 0], label="Policy")
    axes[1].plot(t, states[:, 1])
    horizon = base.max_steps * base.p.dt
    reference_t = np.arange(base.max_steps + 1) * base.p.dt
    full_reference = np.asarray([base.mission(float(ti)) for ti in reference_t])
    full_z_ref = full_reference if full_reference.ndim == 1 else full_reference[:, 0]
    axes[0].step(reference_t, full_z_ref, where="post", color="black",
                 linestyle="--", label="Mission target")
    axes[0].legend()
    status = ", ".join(outcome) if isinstance(outcome, tuple) else outcome
    axes[0].set_title(f"{type(agent).__name__}: {status} at {t[-1]:.2f} s")
    ylabels = ["Altitude z (m)", "Vertical velocity (m/s)"]
    if has_angles:
        for ax, index, name in zip(axes[2:5], [11, 9, 10], ["psi", "theta", "phi"]):
            ax.plot(t, states[:, index], label=f"Environment {name}")
        ylabels.extend(["Yaw psi (rad)", "Pitch theta (rad; uncontrolled)", "Roll phi (rad)"])
        if full_reference.ndim == 2:
            axes[2].plot(reference_t, full_reference[:, -1], "k--", label="Yaw target")
            if full_reference.shape[1] in (3, 4):
                phi_index = 1 if full_reference.shape[1] == 3 else 2
                axes[4].plot(reference_t, full_reference[:, phi_index], "k--", label="Roll target")
            axes[2].legend()
            axes[4].legend()
    for i, name in enumerate(command_names):
        axes[-1].step(t, np.append(actions[:, i], actions[-1, i]), where="post", label=name)
    axes[-1].legend()
    ylabels.append("Raw motor command")
    for ax, ylabel in zip(axes, ylabels):
        ax.set_ylabel(ylabel)
        ax.axvline(t[-1], color="tab:blue", linestyle=":", alpha=0.4)
        ax.grid(alpha=0.3)
    axes[-1].set(xlabel="Time (s)", xlim=(0, horizon))
    fig.tight_layout()
    fig.savefig(output_dir / "flight.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return metrics


def plot_learning_curve(log_dir, output_path, *, show=False):
    """Plot episode returns and deterministic evaluations."""
    import matplotlib.pyplot as plt
    from stable_baselines3.common.results_plotter import load_results

    log_dir = Path(log_dir)
    history = load_results(str(log_dir))
    returns = history["r"].to_numpy()
    steps = history["l"].cumsum().to_numpy()
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
