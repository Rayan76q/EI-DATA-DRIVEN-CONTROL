"""Policy training with saved settings, checkpoints and TensorBoard logs."""

import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy"

import numpy as np
import torch
from stable_baselines3 import DQN, PPO, SAC, __version__ as sb3_version
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from .export_policy import export_for_matlab
from .lib.droneEnv import DroneEnv, Z_LAND


CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_CONFIG = CONFIG_DIR / "ppo.json"
POLICY = "ppo"


def ref_mission(t):
    """Hold at 0 m until 2 s, hover at 2 m until 6 s, then land."""
    return 0.0 if t < 2.0 else (2.0 if t < 6.0 else Z_LAND)


def load_ppo_config(path=DEFAULT_CONFIG):
    """Load training settings from a source config or a saved run config."""
    return json.loads(Path(path).expanduser().read_text())


def train_ppo(
    rnn_model,
    mission,
    output_dir,
    *,
    total_timesteps=None,
    seed=None,
    n_envs=None,
    env_class=DroneEnv,
    initial_log_std=None,
    config_path=DEFAULT_CONFIG,
    **ppo_params,
):
    """Train PPO, export its selected checkpoint, and return the agent/path.

    Each environment has its own recurrent state and shares the RNN weights.
    Use ``lib.droneEnvRotation.DroneEnv`` for the rotation environment.
    Explicit arguments override the config. None uses the configured value.
    """
    config = load_ppo_config(config_path)
    total_timesteps = config["total_timesteps"] if total_timesteps is None else total_timesteps
    seed = config["seed"] if seed is None else seed
    n_envs = config["n_envs"] if n_envs is None else n_envs
    initial_log_std = config["initial_log_std"] if initial_log_std is None else initial_log_std
    params = {**config["ppo"], **ppo_params}
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "total_timesteps": total_timesteps,
        "seed": seed,
        "n_envs": n_envs,
        "initial_log_std": initial_log_std,
        "ppo": params,
        "metadata": {
            "algorithm": "PPO",
            "source_config": str(Path(config_path).expanduser().resolve()),
            "environment": f"{env_class.__module__}.{env_class.__name__}",
            "stable_baselines3": sb3_version,
            "torch": str(torch.__version__),
        },
    }
    config_snapshot = output_dir / "ppo_config.json"
    config_snapshot.write_text(json.dumps(resolved_config, indent=2) + "\n")
    train_env = make_vec_env(
        lambda: env_class(rnn_model=rnn_model, mission=mission),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(output_dir),
    )
    eval_env = make_vec_env(
        lambda: env_class(rnn_model=rnn_model, mission=mission),
        n_envs=1,
        seed=100,
    )
    try:
        agent = PPO(
            "MlpPolicy", train_env, seed=seed, device="cpu", verbose=0,
            tensorboard_log=str(output_dir / "tensorboard"), **params,
        )
        if initial_log_std is not None:
            with torch.no_grad():
                agent.policy.log_std.copy_(torch.tensor(initial_log_std, dtype=torch.float32))

        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(output_dir),
            log_path=str(output_dir),
            eval_freq=max(1, 20_480 // n_envs),
            n_eval_episodes=1,
            deterministic=True,
            verbose=0,
        )
        agent.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name="PPO")
        # SB3 logs before each update; save the final update's losses too.
        agent.logger.dump(step=agent.num_timesteps)
        agent.logger.close()
        final_path = output_dir / "ppo_drone_rnn.zip"
        agent.save(final_path)
        best_path = output_dir / "best_model.zip"
        checkpoint_path = (
            best_path if best_path.is_file() and callback.best_mean_reward > float("-inf") else final_path
        )
        resolved_config["metadata"].update(
            actual_timesteps=agent.num_timesteps,
            checkpoint=str(checkpoint_path.resolve()),
        )
        selected_agent = PPO.load(checkpoint_path, device="cpu")
        export_env = env_class(rnn_model=rnn_model, mission=mission)
        try:
            if export_env.observation_space.shape == (3,) and export_env.action_space.shape == (1,):
                policy_path, trajectory_path = export_for_matlab(selected_agent, export_env, output_dir)
                resolved_config["metadata"]["matlab_export"] = {
                    "status": "exported", "policy": str(policy_path.resolve()),
                    "trajectory": str(trajectory_path.resolve()),
                }
            else:
                resolved_config["metadata"]["matlab_export"] = {
                    "status": "unsupported_environment",
                    "reason": "MATLAB export requires 3 altitude observations and 1 action",
                }
                print("MATLAB export skipped: only the altitude environment is supported")
        finally:
            export_env.close()
        config_snapshot.write_text(json.dumps(resolved_config, indent=2) + "\n")
        return selected_agent, checkpoint_path
    finally:
        train_env.close()
        eval_env.close()


def train_off_policy(
    algorithm,
    rnn_model,
    mission,
    output_dir,
    *,
    total_timesteps=None,
    seed=None,
    n_envs=None,
    env_class=DroneEnv,
    config_path=None,
    **params,
):
    """Train DQN or SAC, export the selected checkpoint, and return agent/path.

    DQN uses the altitude command grid from its config. SAC keeps the
    environment's continuous actions. Explicit arguments override the config.
    """
    algorithm = algorithm.lower()
    if algorithm not in {"dqn", "sac"}:
        raise ValueError("algorithm must be 'dqn' or 'sac'")
    config_path = CONFIG_DIR / f"{algorithm}.json" if config_path is None else config_path
    config = json.loads(Path(config_path).expanduser().read_text())
    total_timesteps = config["total_timesteps"] if total_timesteps is None else total_timesteps
    seed = config["seed"] if seed is None else seed
    n_envs = config["n_envs"] if n_envs is None else n_envs
    params = {**config[algorithm], **params}
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = None
    if algorithm == "dqn":
        from .lib.discrete_actions import DiscreteDroneActions

        grid = config["action_grid"]
        count = grid["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("action_grid.count must be an integer of at least 2")
        if not np.isfinite([grid["low"], grid["high"]]).all() or grid["low"] >= grid["high"]:
            raise ValueError("action_grid requires finite low < high")
        commands = np.linspace(grid["low"], grid["high"], count, dtype=np.float32)
        np.save(output_dir / "commands.npy", commands)

    def make_env():
        env = env_class(rnn_model=rnn_model, mission=mission)
        return env if commands is None else DiscreteDroneActions(env, commands)

    resolved_config = {
        "total_timesteps": total_timesteps,
        "seed": seed,
        "n_envs": n_envs,
        algorithm: params,
        "metadata": {
            "algorithm": algorithm.upper(),
            "source_config": str(Path(config_path).expanduser().resolve()),
            "environment": f"{env_class.__module__}.{env_class.__name__}",
            "stable_baselines3": sb3_version,
            "torch": str(torch.__version__),
        },
    }
    if commands is not None:
        resolved_config["action_grid"] = config["action_grid"]
        resolved_config["metadata"]["commands"] = str((output_dir / "commands.npy").resolve())
    config_snapshot = output_dir / f"{algorithm}_config.json"
    config_snapshot.write_text(json.dumps(resolved_config, indent=2) + "\n")
    train_env = make_vec_env(
        make_env, n_envs=n_envs, seed=seed, monitor_dir=str(output_dir),
    )
    eval_env = make_vec_env(make_env, n_envs=1, seed=100)
    agent = None
    agent_class = DQN if algorithm == "dqn" else SAC
    try:
        agent = agent_class(
            "MlpPolicy", train_env, seed=seed, device="cpu", verbose=0,
            tensorboard_log=str(output_dir / "tensorboard"), **params,
        )
        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(output_dir),
            log_path=str(output_dir),
            eval_freq=max(1, 20_480 // n_envs),
            n_eval_episodes=1,
            deterministic=True,
            verbose=0,
        )
        agent.learn(
            total_timesteps=total_timesteps, callback=callback,
            tb_log_name=algorithm.upper(),
        )
        agent.logger.dump(step=agent.num_timesteps)
        final_path = output_dir / f"{algorithm}_drone_rnn.zip"
        agent.save(final_path)
        best_path = output_dir / "best_model.zip"
        checkpoint_path = (
            best_path if best_path.is_file() and callback.best_mean_reward > float("-inf") else final_path
        )
        resolved_config["metadata"].update(
            actual_timesteps=agent.num_timesteps,
            checkpoint=str(checkpoint_path.resolve()),
        )
        selected_agent = agent_class.load(checkpoint_path, device="cpu")
        export_env = make_env()
        try:
            if export_env.observation_space.shape == (3,) and export_env.unwrapped.action_space.shape == (1,):
                policy_path, trajectory_path = export_for_matlab(selected_agent, export_env, output_dir)
                resolved_config["metadata"]["matlab_export"] = {
                    "status": "exported", "policy": str(policy_path.resolve()),
                    "trajectory": str(trajectory_path.resolve()),
                }
            else:
                resolved_config["metadata"]["matlab_export"] = {
                    "status": "unsupported_environment",
                    "reason": "MATLAB export requires 3 altitude observations and 1 action",
                }
                print("MATLAB export skipped: only the altitude environment is supported")
        finally:
            export_env.close()
        config_snapshot.write_text(json.dumps(resolved_config, indent=2) + "\n")
        return selected_agent, checkpoint_path
    finally:
        if agent is not None and getattr(agent, "_logger", None) is not None:
            agent.logger.close()
        train_env.close()
        eval_env.close()


def main():
    """Train the selected policy and evaluate it on the altitude mission."""
    import argparse
    from datetime import datetime
    import time

    from model.rnn_loader import load_rnn_model
    from .evaluation import evaluate_policy, plot_learning_curve

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", "--algo", choices=("ppo", "dqn", "sac"), default=POLICY,
                        help=f"Policy algorithm to train (default: {POLICY})")
    parser.add_argument("--rnn", type=Path,
                        default=project_root / "src" / "model" / "checkpoints" / "rnn_z.pt",
                        help="Altitude dynamics checkpoint")
    parser.add_argument("--config", type=Path, help="Training settings JSON (default: policy/config/<policy>.json)")
    parser.add_argument("--steps", type=int, help="Training timesteps; overrides the config")
    parser.add_argument("--seed", type=int, help="Random seed; overrides the config")
    parser.add_argument("--output-dir", type=Path, help="Directory for the training run and results")
    parser.add_argument("--show", action="store_true", help="Also display the saved plots")
    args = parser.parse_args()
    config_path = (args.config or CONFIG_DIR / f"{args.policy}.json").expanduser().resolve()
    try:
        config = json.loads(config_path.read_text())
        if args.policy not in config:
            parser.error(f"{config_path} has no '{args.policy}' training settings")
        steps = config["total_timesteps"] if args.steps is None else args.steps
        seed = config["seed"] if args.seed is None else args.seed
        if steps <= 0:
            parser.error("Training timesteps must be positive")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    torch.set_num_threads(1)

    output_dir = args.output_dir or (
        project_root / "results" / "altitude" / args.policy / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = output_dir / "training"
    rnn_path = args.rnn.expanduser().resolve()
    rnn_model = load_rnn_model(rnn_path)
    print(f"RNN: {rnn_path}", flush=True)
    print("Mission: 0 m until 2 s, 2 m until 6 s, then land at -0.3 m", flush=True)
    print(f"{args.policy.upper()} training: {steps:,} requested steps", flush=True)
    started = time.perf_counter()
    training_args = dict(total_timesteps=steps, seed=seed, config_path=config_path)
    if args.policy == "ppo":
        agent, checkpoint_path = train_ppo(rnn_model, ref_mission, training_dir, **training_args)
    else:
        agent, checkpoint_path = train_off_policy(
            args.policy, rnn_model, ref_mission, training_dir, **training_args,
        )
    print(f"Training completed in {time.perf_counter() - started:.1f} s; selected {checkpoint_path}", flush=True)
    env = DroneEnv(rnn_model=rnn_model, mission=ref_mission)
    try:
        if args.policy == "dqn":
            from .lib.discrete_actions import DiscreteDroneActions
            env = DiscreteDroneActions(env, np.load(training_dir / "commands.npy", allow_pickle=False))
        metrics = evaluate_policy(agent, env, output_dir, seed=seed, show=args.show)
        metrics.update(
            algorithm=args.policy.upper(), rnn_path=str(rnn_path), policy_path=str(checkpoint_path),
            seed=seed, Ts=float(env.unwrapped.p.dt), policy_training_steps=int(agent.num_timesteps),
            trained_this_run=True, training_config_path=str(config_path),
            mission={"initial_altitude_m": 0.0, "hover_altitude_m": 2.0, "climb_start_s": 2.0,
                     "landing_start_s": 6.0, "landing_altitude_m": Z_LAND},
        )
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        snapshot_path = training_dir / f"{args.policy}_config.json"
        snapshot = json.loads(snapshot_path.read_text())
        snapshot["metadata"].update(rnn_path=str(rnn_path), mission=metrics["mission"], results=str(output_dir))
        snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
        plot_learning_curve(training_dir, output_dir / "learning_curve.png", show=args.show)
        print(f"Result: {metrics['outcome']} at {metrics['duration_s']:.2f} s | "
              f"peak {metrics['peak_altitude_m']:.3f} m | "
              f"final vz {metrics['final_vertical_velocity_m_s']:.3f} m/s | "
              f"return {metrics['return']:.1f}")
        print(f"Results: {output_dir}")
        print(f"TensorBoard: tensorboard --logdir {training_dir / 'tensorboard'}")
        matlab_dir = str(project_root / "src" / "drone" / "closed_loop").replace("'", "''")
        matlab_output = str(training_dir).replace("'", "''")
        print(f"MATLAB: addpath('{matlab_dir}'); EINT_RL_eval('{matlab_output}', 0.0)")
    finally:
        env.close()


if __name__ == "__main__":
    main()

