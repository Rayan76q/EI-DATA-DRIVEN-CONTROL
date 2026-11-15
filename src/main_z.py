"""Evaluate or train an altitude policy, then export and run it in Simulink."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO, SAC

from model.rnn_loader import load_rnn_model
from policy.evaluation import evaluate_policy, plot_learning_curve
from policy.export_policy import export_for_matlab
from policy.lib.discrete_actions import DiscreteDroneActions
from policy.lib.droneEnv import DroneEnv, Z_LAND
from policy.matlab import evaluate_export
from policy.training import (
    CONFIG_DIR,
    CurriculumCallback,
    configured_checkpoint,
    ref_mission,
    train_off_policy,
    train_ppo,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {"ppo": PPO, "dqn": DQN, "sac": SAC}


def parse_arguments(argv):
    """Keep evaluation defaults in the same JSON files used for training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-model", choices=ALGORITHMS, default="ppo")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--train", action="store_true")
    selection.add_argument("--policy", type=Path, help="Override the configured saved policy")
    parser.add_argument("--rnn", type=Path, default=PROJECT_ROOT / "src/model/checkpoints/rnn_z.pt")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int, help="Override the training budget")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--from-scratch", action="store_true", help="Train SAC without its starting checkpoint",
    )
    parser.add_argument(
        "--curriculum", type=int, metavar="K", help="Start PPO/SAC at curriculum height index K",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--matlab", type=Path, help="Override MATLAB executable discovery")
    parser.add_argument("--skip-matlab", action="store_true")
    args = parser.parse_args(argv)

    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.from_scratch and (not args.train or args.rl_model != "sac"):
        parser.error("--from-scratch requires --train --rl-model sac")
    if args.curriculum is not None and (not args.train or args.rl_model == "dqn"):
        parser.error("--curriculum requires PPO or SAC training")
    return args


def train_agent(args, model, output_dir, config_path, seed):
    """Train into a new run directory without replacing the selected reference policy."""
    settings = dict(total_timesteps=args.steps, seed=seed, config_path=config_path)
    if args.curriculum is not None:
        settings["curriculum"] = CurriculumCallback(difficulty=args.curriculum, verbose=1)
    if args.rl_model == "ppo":
        return train_ppo(model, ref_mission, output_dir, **settings)
    return train_off_policy(
        args.rl_model, model, ref_mission, output_dir,
        from_scratch=args.from_scratch, **settings,
    )


def main(argv=None):
    """Run RNN evaluation, MATLAB export and optional physical-plant validation."""
    args = parse_arguments(argv)
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    torch.set_num_threads(1)

    config_path = (args.config or CONFIG_DIR / f"{args.rl_model}.json").expanduser().resolve()
    config = json.loads(config_path.read_text())
    seed = config["seed"] if args.seed is None else args.seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or PROJECT_ROOT / "results/runs/z" / args.rl_model / timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rnn_path = args.rnn.expanduser().resolve()
    model = load_rnn_model(rnn_path)

    if args.train:
        agent, checkpoint = train_agent(args, model, output_dir / "training", config_path, seed)
        plot_learning_curve(output_dir / "training", output_dir / "learning_curve.png", show=args.show)
    else:
        checkpoint = args.policy or configured_checkpoint(config)
        if checkpoint is None:
            raise ValueError("Set checkpoint in the config or pass --policy PATH")
        checkpoint = checkpoint.expanduser().resolve()
        agent = ALGORITHMS[args.rl_model].load(checkpoint, device="cpu")

    env = DroneEnv(rnn_model=model, mission=ref_mission)
    if args.rl_model == "dqn":
        commands = np.load(checkpoint.parent / "commands.npy", allow_pickle=False)
        env = DiscreteDroneActions(env, commands)

    try:
        if agent.observation_space != env.observation_space or agent.action_space != env.action_space:
            raise ValueError("Policy observations/actions do not match the altitude environment")
        metrics = evaluate_policy(agent, env, output_dir, seed=seed, show=args.show)
        export_for_matlab(agent, env, output_dir)
        metrics.update(
            algorithm=args.rl_model.upper(), rnn_path=str(rnn_path),
            policy_path=str(checkpoint), seed=seed, Ts=float(env.unwrapped.p.dt),
            policy_training_steps=int(agent.num_timesteps), trained_this_run=args.train,
            mission=dict(initial_altitude_m=0., hover_altitude_m=2., climb_start_s=2.,
                         landing_start_s=6., landing_altitude_m=Z_LAND),
        )
        metrics["simulink"] = {"status": "skipped"}
        if not args.skip_matlab:
            metrics["simulink"] = evaluate_export(output_dir, output_dir, executable=args.matlab)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Policy: {checkpoint}")
        print(f"Result: {metrics['outcome']} at {metrics['duration_s']:.2f} s")
        print(f"MATLAB/Simulink: {metrics['simulink']['status']}")
        print(f"Results: {output_dir}")
        if metrics["simulink"]["status"] in {"failed", "mission_failed"}:
            raise RuntimeError(f"Simulink validation failed; see {output_dir / 'matlab.log'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
