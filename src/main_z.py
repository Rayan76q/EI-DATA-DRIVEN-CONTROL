"""Climb to 2 m, hover, then land."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.env_checker import check_env

from model.rnn_loader import load_rnn_model
from policy.lib.droneEnv import DroneEnv, Z_LAND
from policy.lib.discrete_actions import DiscreteDroneActions
from policy.evaluation import evaluate_policy, plot_learning_curve
from policy.export_policy import export_for_matlab
from policy.training import CONFIG_DIR, CurriculumCallback, ref_mission, train_off_policy, train_ppo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RL_MODEL = "ppo"  # Select "ppo", "dqn" or "sac" for evaluation.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-model", "--algo", dest="algo", choices=("ppo", "dqn", "sac"),
                        default=RL_MODEL, help="RL model to evaluate (default: RL_MODEL in main_z.py)")
    policy_selection = parser.add_mutually_exclusive_group()
    policy_selection.add_argument("--train", action="store_true", help="Train a new policy instead of loading a saved one")
    policy_selection.add_argument("--policy", type=Path, help="Load a checkpoint for the selected algorithm to evaluate and export")
    parser.add_argument("--rnn", type=Path,
                        default=PROJECT_ROOT / "src" / "model" / "checkpoints" / "rnn_z.pt",
                        help="Altitude dynamics checkpoint (default: model/checkpoints/rnn_z.pt)")
    parser.add_argument("--config", type=Path, help="Training settings JSON (default: policy/config/<algo>.json)")
    parser.add_argument("--steps", type=int, help="Training timesteps; overrides the config")
    parser.add_argument("--curriculum", type=int, nargs="?", const=0, metavar="K",
                        help="With --train: cycle z heights, starting at index K (default: 0)")
    parser.add_argument("--seed", type=int, help="Random seed; overrides the config")
    parser.add_argument("--output-dir", type=Path, help="Directory for plots, metrics, trajectories and MATLAB export")
    parser.add_argument("--show", action="store_true", help="Also display the saved plots")
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.curriculum is not None and (not args.train or args.algo == "dqn"):
        parser.error("--curriculum requires --train with PPO or SAC")
    config_path = (args.config or CONFIG_DIR / f"{args.algo}.json").expanduser().resolve()
    try:
        config = json.loads(config_path.read_text())
        if args.algo not in config:
            parser.error(f"{config_path} has no '{args.algo}' training settings")
        steps = args.steps if args.steps is not None else config["total_timesteps"]
        seed = args.seed if args.seed is not None else config["seed"]
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    agent_class = {"ppo": PPO, "dqn": DQN, "sac": SAC}[args.algo]
    if not args.train:
        checkpoint_path = args.policy
        if checkpoint_path is None:
            runs = sorted(
                (PROJECT_ROOT / "results" / "altitude" / args.algo).glob(f"*/training/{args.algo}_config.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for run_config_path in runs:
                try:
                    run_config = json.loads(run_config_path.read_text())
                except (OSError, ValueError):
                    continue
                metadata = run_config.get("metadata") if isinstance(run_config, dict) else None
                saved_checkpoint = metadata.get("checkpoint") if isinstance(metadata, dict) else None
                if isinstance(saved_checkpoint, str) and Path(saved_checkpoint).is_file():
                    checkpoint_path = Path(saved_checkpoint)
                    break
        checkpoint_path = (checkpoint_path or (
            PROJECT_ROOT / "src" / "policy" / "checkpoints" / f"{args.algo}_drone_rnn.zip"
        )).expanduser().resolve()
        if not checkpoint_path.is_file() and Path(f"{checkpoint_path}.zip").is_file():
            checkpoint_path = Path(f"{checkpoint_path}.zip")
        if not checkpoint_path.is_file():
            parser.error(f"Checkpoint not found: {checkpoint_path}. "
                         f"Train it with python src/policy/training.py --policy {args.algo}, or use --policy PATH.")
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    torch.set_num_threads(1)

    output_dir = args.output_dir or (
        PROJECT_ROOT / "results" / "altitude" / args.algo / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rnn_path = args.rnn.expanduser().resolve()
    rnn_model = load_rnn_model(rnn_path)
    print(f"RNN: {rnn_path}")
    print("Mission: 0 m until 2 s, 2 m until 6 s, then land at -0.3 m")

    env = DroneEnv(rnn_model=rnn_model, mission=ref_mission)
    try:
        check_env(env)
        if args.train:
            print(f"{args.algo.upper()} training: {steps:,} requested steps")
            started = time.perf_counter()
            training_args = dict(total_timesteps=steps, seed=seed, config_path=config_path)
            if args.curriculum is not None:
                training_args["curriculum"] = CurriculumCallback(difficulty=args.curriculum, verbose=1)
            if args.algo == "ppo":
                agent, checkpoint_path = train_ppo(
                    rnn_model, ref_mission, output_dir / "training", **training_args,
                )
            else:
                agent, checkpoint_path = train_off_policy(
                    args.algo, rnn_model, ref_mission, output_dir / "training", **training_args,
                )
            print(f"Selected checkpoint: {agent.num_timesteps:,} steps; {time.perf_counter() - started:.1f} s")
            print(f"TensorBoard: tensorboard --logdir {output_dir / 'training' / 'tensorboard'}")
            plot_learning_curve(output_dir / "training", output_dir / "learning_curve.png", show=args.show)
        else:
            agent = agent_class.load(checkpoint_path, device="cpu")

        if args.algo == "dqn":
            commands_path = checkpoint_path.parent / "commands.npy"
            if not commands_path.is_file():
                raise FileNotFoundError(f"DQN needs its command grid beside the checkpoint: {commands_path}")
            env = DiscreteDroneActions(env, np.load(commands_path, allow_pickle=False))

        if agent.observation_space != env.observation_space or agent.action_space != env.action_space:
            raise ValueError("The selected policy does not match the altitude environment's observations/actions")
        print(f"Policy: {checkpoint_path}")
        metrics = evaluate_policy(agent, env, output_dir, seed=seed, show=args.show)
        metrics.update(
            algorithm=args.algo.upper(),
            rnn_path=str(rnn_path),
            policy_path=str(checkpoint_path),
            seed=seed,
            Ts=float(env.unwrapped.p.dt),
            policy_training_steps=int(agent.num_timesteps),
            trained_this_run=args.train,
            training_config_path=str(config_path) if args.train else None,
            mission={
                "initial_altitude_m": 0.0,
                "hover_altitude_m": 2.0,
                "climb_start_s": 2.0,
                "landing_start_s": 6.0,
                "landing_altitude_m": Z_LAND,
            },
        )
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        export_for_matlab(agent, env, output_dir)
        print(f"Result: {metrics['outcome']} at {metrics['duration_s']:.2f} s | "
              f"peak {metrics['peak_altitude_m']:.3f} m | "
              f"final vz {metrics['final_vertical_velocity_m_s']:.3f} m/s | "
              f"return {metrics['return']:.1f}")
        print(f"Results: {output_dir}")
        matlab_dir = str(PROJECT_ROOT / "src" / "drone" / "closed_loop").replace("'", "''")
        matlab_output = str(output_dir).replace("'", "''")
        matlab_command = f"addpath('{matlab_dir}'); EINT_RL_eval('{matlab_output}', 0.0)"
        matlab = shutil.which("matlab")
        if matlab is None:
            metrics["simulink"] = {"status": "unavailable", "reason": "MATLAB is not installed or not on PATH"}
            print("Simulink not run: MATLAB is not installed or not on PATH.")
            print(f"MATLAB: {matlab_command}")
        else:
            matlab_log = output_dir / "matlab.log"
            print("Running the Simulink plant; scope exports will be saved under results/closed_loop.", flush=True)
            with matlab_log.open("w") as log:
                result = subprocess.run(
                    [matlab, "-batch", matlab_command], cwd=PROJECT_ROOT,
                    stdout=log, stderr=subprocess.STDOUT,
                )
            metrics["simulink"] = {
                "status": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode, "log": str(matlab_log),
            }
            print(f"Simulink: {metrics['simulink']['status']} | log: {matlab_log}")
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if metrics["simulink"]["status"] == "failed":
            raise RuntimeError(f"Simulink failed; see {metrics['simulink']['log']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
