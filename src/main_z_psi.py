"""Evaluate or train a PPO policy for altitude landing with a full yaw turn."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from model.rnn_loader import load_rnn_model
from policy.evaluation import evaluate_rotation, plot_learning_curve
from policy.export_policy import export_for_matlab
from policy.lib.droneEnvRotation import DroneEnv, default_mission
from policy.matlab import evaluate_export
from policy.training import (
    CONFIG_DIR,
    ScaledControlActions,
    configured_checkpoint,
    dynamics_fingerprint,
    load_ppo_config,
    train_ppo,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments(argv):
    """Select the shared config, saved policy and optional training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--train", action="store_true")
    selection.add_argument("--policy", type=Path, help="Override the configured saved policy")
    parser.add_argument("--resume", type=Path, help="Continue PPO training from this checkpoint")
    parser.add_argument("--steps", type=int, help="Override the total training budget")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rnn", type=Path, default=PROJECT_ROOT / "src/model/checkpoints/rnn_z_psi.pt")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "ppo_z_psi.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--matlab", type=Path, help="Override MATLAB executable discovery")
    parser.add_argument("--skip-matlab", action="store_true")
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.resume is not None and not args.train:
        parser.error("--resume requires --train")
    return args


def main(argv=None):
    """Use the saved scaling and verify dynamics before flying the full-turn mission."""
    args = parse_arguments(argv)
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    torch.set_num_threads(1)

    config = load_ppo_config(args.config)
    seed = config["seed"] if args.seed is None else args.seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or PROJECT_ROOT / "results/runs/z_psi" / timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rnn_path = args.rnn.expanduser().resolve()
    model = load_rnn_model(rnn_path)

    if args.train:
        agent, checkpoint = train_ppo(
            model, default_mission, output_dir / "training", env_class=DroneEnv,
            total_timesteps=args.steps, seed=seed, config_path=args.config,
            initial_checkpoint=args.resume,
        )
        plot_learning_curve(output_dir / "training", output_dir / "learning_curve.png", show=args.show)
    else:
        checkpoint = args.policy or configured_checkpoint(config)
        if checkpoint is None:
            raise ValueError("Set checkpoint in the config or pass --policy PATH")
        checkpoint = checkpoint.expanduser().resolve()
        agent = PPO.load(checkpoint, device="cpu")

    env = DroneEnv(rnn_model=model, mission=default_mission)
    if getattr(agent, "action_scale", None) is not None:
        env = ScaledControlActions(env, agent.action_scale, agent.action_offset)
    try:
        if agent.observation_space != env.observation_space or agent.action_space != env.action_space:
            raise ValueError("Policy must have six observations and two controls")
        fingerprint = dynamics_fingerprint(model, env.unwrapped.p)
        if getattr(agent, "dynamics_sha256", fingerprint) != fingerprint:
            raise ValueError("Policy was trained with different RNN weights or normalization")

        metrics = evaluate_rotation(agent, env, output_dir, seed=seed, show=args.show)
        export_for_matlab(agent, env, output_dir)
        metrics.update(
            algorithm="PPO", rnn_path=str(rnn_path), policy_path=str(checkpoint),
            seed=seed, policy_training_steps=int(agent.num_timesteps), dynamics_sha256=fingerprint,
        )
        metrics["simulink"] = {"status": "skipped"}
        if not args.skip_matlab:
            metrics["simulink"] = evaluate_export(output_dir, output_dir, executable=args.matlab)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Policy: {checkpoint}")
        print(f"Result: {metrics['outcome']}")
        print(f"MATLAB/Simulink: {metrics['simulink']['status']}")
        print(f"Results: {output_dir}")
        if metrics["simulink"]["status"] in {"failed", "mission_failed"}:
            raise RuntimeError(f"Simulink validation failed; see {output_dir / 'matlab.log'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
