"""Run altitude and yaw control; save results under results/altitude_psi."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from stable_baselines3 import PPO

from model.rnn_loader import load_rnn_model
from policy.lib.droneEnvRotation import DroneEnv, default_mission, run_episode
from policy.training import ScaledControlActions, dynamics_fingerprint, load_ppo_config, raw_action, train_ppo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RNN_PATH = PROJECT_ROOT / "src/model/checkpoints/rnn_model_thrust_speeds.pt"
POLICY_PATH = PROJECT_ROOT / "src/policy/checkpoints/ppo_drone_thrust_speeds.zip"
PPO_CONFIG_PATH = PROJECT_ROOT / "src/policy/config/ppo_z_psi.json"


def save_result(agent, env, output_dir, *, seed=0, show=False):
    """Evaluate one mission and save only altitude, yaw and their commands."""
    import matplotlib.pyplot as plt

    states, actions, total, outcome = run_episode(
        env, lambda obs: agent.predict(obs, deterministic=True)[0], seed=seed,
    )
    actions = np.asarray([raw_action(agent, action) for action in actions])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--train", action="store_true", help="Train a new PPO policy")
    selection.add_argument("--policy", type=Path, default=POLICY_PATH, help="PPO checkpoint to evaluate")
    parser.add_argument("--resume", type=Path, help="With --train: continue this PPO checkpoint to --steps total")
    parser.add_argument("--steps", type=int, help="Training steps; defaults to policy/config/ppo_z_psi.json")
    parser.add_argument("--seed", type=int, default=load_ppo_config(PPO_CONFIG_PATH)["seed"])
    parser.add_argument("--output-dir", type=Path, help="Override results/altitude_psi/<timestamp>")
    parser.add_argument("--show", action="store_true", help="Also display the saved plot")
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.resume is not None and not args.train:
        parser.error("--resume requires --train")
    checkpoint = args.policy.expanduser().resolve()
    if not args.train and not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}. Use --train or --policy PATH.")
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    torch.set_num_threads(1)

    output_dir = (args.output_dir or (
        PROJECT_ROOT / "results/altitude_psi"
        / datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y%m%d_%H%M%S_%f")
    )).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rnn = load_rnn_model(RNN_PATH)
    if args.train:
        agent, checkpoint = train_ppo(
            rnn, default_mission, output_dir / "training", env_class=DroneEnv,
            total_timesteps=args.steps, seed=args.seed,
            config_path=PPO_CONFIG_PATH,
            initial_checkpoint=args.resume,
        )
    else:
        agent = PPO.load(checkpoint, device="cpu")
    env = DroneEnv(rnn_model=rnn, mission=default_mission)
    if getattr(agent, "action_scale", None) is not None:
        env = ScaledControlActions(env, agent.action_scale, agent.action_offset)
    try:
        if agent.observation_space != env.observation_space or agent.action_space != env.action_space:
            raise ValueError("Policy must use the altitude/yaw environment's 6 observations and 2 actions")
        fingerprint = getattr(agent, "dynamics_sha256", None)
        if fingerprint is not None and fingerprint != dynamics_fingerprint(rnn, env.unwrapped.p):
            raise ValueError("The PPO checkpoint was trained with different RNN weights or normalization; retrain it")
        metrics = save_result(agent, env, output_dir, seed=args.seed, show=args.show)
        metrics.update(
            algorithm="PPO", rnn_path=str(RNN_PATH), policy_path=str(checkpoint),
            seed=args.seed, policy_training_steps=int(agent.num_timesteps),
            dynamics_sha256=dynamics_fingerprint(rnn, env.unwrapped.p),
        )
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Result: {metrics['outcome']} | return {metrics['return']:.1f}")
        print(f"Policy: {checkpoint}")
        print(f"Results: {output_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
