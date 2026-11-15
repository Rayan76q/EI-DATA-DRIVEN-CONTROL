"""Policy training with saved settings, checkpoints and TensorBoard logs."""

import json
import hashlib
from collections import deque
from pathlib import Path

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy"

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import DQN, PPO, SAC, __version__ as sb3_version
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .export_policy import export_for_matlab
from .lib.droneEnv import CURRICULUM_HEIGHTS, DroneEnv, Z_LAND


CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_CONFIG = CONFIG_DIR / "ppo.json"
POLICY = "ppo"


class ScaledControlActions(gym.ActionWrapper):
    """Let PPO use [-1, 1] while the environment receives physical commands."""

    def __init__(self, env, scale, offset):
        super().__init__(env)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.offset = np.asarray(offset, dtype=np.float32)
        if self.scale.shape != env.action_space.shape or self.offset.shape != self.scale.shape:
            raise ValueError("action_scale and action_offset must match the action shape")
        if not np.all(self.scale > 0):
            raise ValueError("action_scale must be positive")
        if np.any(self.offset - self.scale < env.action_space.low) or np.any(self.offset + self.scale > env.action_space.high):
            raise ValueError("Scaled action range must fit within the raw environment bounds")
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=self.scale.shape, dtype=np.float32)

    def action(self, action):
        return np.clip(self.offset + self.scale * np.asarray(action, dtype=np.float32),
                       self.env.action_space.low, self.env.action_space.high)


def raw_action(agent, action):
    """Convert a saved PPO action to the raw environment's thrust/yaw units."""
    if getattr(agent, "action_scale", None) is None:
        return action
    return (np.asarray(agent.action_offset, dtype=np.float32)
            + np.asarray(agent.action_scale, dtype=np.float32) * np.asarray(action, dtype=np.float32))


class ScaledObservation(BaseFeaturesExtractor):
    """Fixed physical-unit scaling saved inside the PPO policy checkpoint."""

    def __init__(self, observation_space, scale):
        super().__init__(observation_space, int(np.prod(observation_space.shape)))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        if self.scale.shape != observation_space.shape or torch.any(self.scale <= 0):
            raise ValueError("observation_scale must contain one positive value per observation")

    def forward(self, observations):
        return observations / self.scale


class SuccessEvalCallback(EvalCallback):
    """Keep successful missions ahead of failed missions when selecting a policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_success_rank = (0.0, -float("inf"))

    def _on_step(self):
        is_evaluation = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        keep_training = super()._on_step()
        if is_evaluation and self._is_success_buffer:
            rank = (float(np.mean(self._is_success_buffer)), self.last_mean_reward)
            if rank[0] > 0 and rank > self.best_success_rank:
                self.best_success_rank = rank
                self.model.save(Path(self.best_model_save_path) / "best_success.zip")
        return keep_training

    def evaluate_now(self):
        frequency = self.eval_freq
        self.eval_freq = 1
        try:
            return self._on_step()
        finally:
            self.eval_freq = frequency

    def _on_training_start(self):
        self.evaluate_now()


def dynamics_fingerprint(model, params=None):
    """Identify the exact learned weights, independent of checkpoint packaging."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    if params is not None:
        values = {key: np.asarray(value).tolist() for key, value in vars(params).items()}
        digest.update(json.dumps(values, sort_keys=True).encode())
    return digest.hexdigest()


def configured_checkpoint(config, key="checkpoint"):
    """Resolve a configured policy path relative to the project root."""
    checkpoint = config.get(key)
    if checkpoint is None:
        return None
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = CONFIG_DIR.parents[2] / path
    return path.resolve()


class _WarmStartEvalCallback(EvalCallback):
    """Include the starting policy in selection before any learning occurs."""

    def _on_training_start(self):
        self._on_step()
        self.initial_mean_reward = self.last_mean_reward


class CurriculumCallback(BaseCallback):
    """Cycle z heights after 85% success over 20 completed episodes.

    ``difficulty`` is the starting index k in [0.5, 1, 2, 5, 10]. Environments
    apply changes at reset. Success means reaching the target, then landing softly.
    For direct SB3 use, initialize DroneEnv(difficulty=callback.difficulty).
    """

    heights = CURRICULUM_HEIGHTS

    def __init__(self, difficulty=0, success_threshold=0.7, window_size=30, verbose=0):
        super().__init__(verbose)
        if not isinstance(difficulty, (int, np.integer)):
            raise TypeError("difficulty must be an integer height index")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError("success_threshold must be between 0 and 1")
        self.difficulty = int(difficulty) % len(self.heights)
        self.success_threshold = success_threshold
        self.successes = deque(maxlen=window_size)

    def _on_training_start(self):
        self.successes.clear()
        self.training_env.env_method("set_difficulty", self.difficulty)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            # Ignore episodes still finishing the previous difficulty.
            if done and info.get("difficulty") == self.difficulty:
                self.successes.append(bool(info["is_success"]))
        return True

    def _on_rollout_end(self):
        self.logger.record("curriculum/difficulty", self.difficulty)
        self.logger.record("curriculum/height", self.heights[self.difficulty])
        if not self.successes:
            return
        success_rate = float(np.mean(self.successes))
        self.logger.record("curriculum/success_rate", success_rate)
        if len(self.successes) == self.successes.maxlen and success_rate >= self.success_threshold:
            self.difficulty = (self.difficulty + 1) % len(self.heights)
            self.successes.clear()
            self.training_env.env_method("set_difficulty", self.difficulty)
            if self.verbose:
                print(f"Curriculum: k={self.difficulty}, height={self.heights[self.difficulty]:g} m")


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
    initial_checkpoint=None,
    config_path=DEFAULT_CONFIG,
    curriculum=None,
    **ppo_params,
):
    """Train PPO, export its selected checkpoint, and return the agent/path.

    Each environment has its own recurrent state and shares the RNN weights.
    Use ``lib.droneEnvRotation.DroneEnv`` for the rotation environment.
    Explicit arguments override the config. None uses the configured value.
    Pass CurriculumCallback(...) to enable the altitude curriculum.
    """
    config = load_ppo_config(config_path)
    total_timesteps = config["total_timesteps"] if total_timesteps is None else total_timesteps
    seed = config["seed"] if seed is None else seed
    n_envs = config["n_envs"] if n_envs is None else n_envs
    initial_log_std = config["initial_log_std"] if initial_log_std is None else initial_log_std
    initial_action = config.get("initial_action")
    observation_scale = config.get("observation_scale")
    action_scale = config.get("action_scale")
    action_offset = config.get("action_offset")
    initial_checkpoint = initial_checkpoint or configured_checkpoint(config, "initial_checkpoint")
    params = {**config["ppo"], **ppo_params}
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if curriculum is not None and not issubclass(env_class, DroneEnv):
        raise ValueError("Curriculum requires the z-only DroneEnv")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "total_timesteps": total_timesteps,
        "seed": seed,
        "n_envs": n_envs,
        "initial_log_std": initial_log_std,
        "initial_action": initial_action,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "observation_scale": observation_scale,
        "action_scale": action_scale,
        "action_offset": action_offset,
        "select_by_success": config.get("select_by_success", False),
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
    snapshot_env = env_class(rnn_model=rnn_model, mission=mission)
    try:
        resolved_config["metadata"]["dynamics"] = {
            "weights_sha256": dynamics_fingerprint(rnn_model),
            "parameters": {key: np.asarray(value).tolist() for key, value in vars(snapshot_env.p).items()},
            "mission": [np.asarray(mission(i * snapshot_env.p.dt)).tolist()
                        for i in range(snapshot_env.max_steps + 1)],
        }
        resolved_config["metadata"]["dynamics_sha256"] = dynamics_fingerprint(rnn_model, snapshot_env.p)
        if config.get("select_by_success"):
            from .lib.droneEnvRotation import REWARD_PARAMS
            resolved_config["metadata"]["reward"] = vars(REWARD_PARAMS).copy()
    finally:
        snapshot_env.close()
    if curriculum is not None:
        resolved_config["curriculum"] = {
            "difficulty": curriculum.difficulty, "heights": list(curriculum.heights),
            "success_threshold": curriculum.success_threshold,
            "window_size": curriculum.successes.maxlen,
        }
    config_snapshot.write_text(json.dumps(resolved_config, indent=2) + "\n")
    def make_training_env(training=False):
        env = env_class(rnn_model=rnn_model, mission=mission, **(
            {"difficulty": curriculum.difficulty} if training and curriculum is not None else {}
        ))
        return ScaledControlActions(env, action_scale, action_offset) if action_scale is not None else env

    train_env = make_vec_env(
        lambda: make_training_env(training=True),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(output_dir),
    )
    eval_env = make_vec_env(
        make_training_env,
        n_envs=1,
        seed=100,
    )
    try:
        runtime_params = dict(params)
        if observation_scale is not None:
            runtime_params["policy_kwargs"] = {
                **(params.get("policy_kwargs") or {}),
                "features_extractor_class": ScaledObservation,
                "features_extractor_kwargs": {"scale": observation_scale},
            }
        if initial_checkpoint is not None:
            agent = PPO.load(
                initial_checkpoint, env=train_env, device="cpu",
                tensorboard_log=str(output_dir / "tensorboard"),
            )
            # PPO.load restores its own settings; reject conflicting resume configs.
            for name, requested in params.items():
                actual = getattr(agent, name)
                if name == "policy_kwargs":
                    matches = all(actual.get(key) == value for key, value in requested.items())
                else:
                    actual = actual(1.0) if callable(actual) else actual
                    matches = actual == requested
                if not matches:
                    raise ValueError(f"Resumed PPO has a different {name}; use its saved training config")
            agent.set_random_seed(seed)
        else:
            agent = PPO(
                "MlpPolicy", train_env, seed=seed, device="cpu", verbose=0,
                tensorboard_log=str(output_dir / "tensorboard"), **runtime_params,
            )
        if initial_action is not None and initial_checkpoint is None:
            with torch.no_grad():
                action = torch.as_tensor(initial_action, dtype=torch.float32)
                if action.shape != agent.policy.action_net.bias.shape:
                    raise ValueError("initial_action must match the environment action shape")
                agent.policy.action_net.weight.zero_()
                agent.policy.action_net.bias.copy_(action)
        if initial_log_std is not None and initial_checkpoint is None:
            with torch.no_grad():
                agent.policy.log_std.copy_(torch.tensor(initial_log_std, dtype=torch.float32))

        if initial_checkpoint is not None:
            if getattr(agent, "action_scale", None) != action_scale or getattr(agent, "action_offset", None) != action_offset:
                raise ValueError("Resumed PPO must use the same action scaling as this config")
            previous_dynamics = getattr(agent, "dynamics_sha256", None)
            if previous_dynamics is not None and previous_dynamics != resolved_config["metadata"]["dynamics_sha256"]:
                raise ValueError("Resumed PPO used different RNN weights or normalization")
            saved_scale = getattr(agent.policy.features_extractor, "scale", None)
            if ((saved_scale is None) != (observation_scale is None) or (
                observation_scale is not None and not np.allclose(
                    saved_scale.detach().cpu().numpy(), observation_scale
                )
            )):
                raise ValueError("Resumed PPO must use the same observation scaling as this config")
        agent.action_scale, agent.action_offset = action_scale, action_offset
        if config.get("select_by_success"):
            agent.dynamics_sha256 = resolved_config["metadata"]["dynamics_sha256"]

        callback_class = SuccessEvalCallback if config.get("select_by_success") else EvalCallback
        callback = callback_class(
            eval_env,
            best_model_save_path=str(output_dir),
            log_path=str(output_dir),
            eval_freq=max(1, 20_480 // n_envs),
            n_eval_episodes=1,
            deterministic=True,
            verbose=0,
        )
        callbacks = callback if curriculum is None else [curriculum, callback]
        initial_steps = agent.num_timesteps
        if initial_steps >= total_timesteps:
            raise ValueError("total_timesteps must exceed the resumed policy's existing steps")
        if initial_checkpoint is not None:
            resolved_config["metadata"]["warm_start"] = {
                "checkpoint": str(initial_checkpoint), "initial_timesteps": initial_steps,
            }
        agent.learn(
            total_timesteps=total_timesteps - initial_steps, callback=callbacks,
            tb_log_name="PPO", reset_num_timesteps=initial_checkpoint is None,
        )
        if isinstance(callback, SuccessEvalCallback):
            callback.evaluate_now()
        # SB3 logs before each update; save the final update's losses too.
        agent.logger.dump(step=agent.num_timesteps)
        agent.logger.close()
        final_path = output_dir / "ppo_drone_rnn.zip"
        agent.save(final_path)
        best_path = output_dir / "best_model.zip"
        success_path = output_dir / "best_success.zip"
        # Fixed 2 m evaluation cannot select progress across curriculum heights.
        checkpoint_path = success_path if config.get("select_by_success") and success_path.is_file() else (
            best_path if curriculum is None and best_path.is_file()
            and callback.best_mean_reward > float("-inf") else final_path
        )
        resolved_config["metadata"].update(
            actual_timesteps=agent.num_timesteps,
            checkpoint=str(checkpoint_path.resolve()),
        )
        selected_agent = PPO.load(checkpoint_path, device="cpu")
        if config.get("select_by_success"):
            selected_agent.dynamics_sha256 = resolved_config["metadata"]["dynamics_sha256"]
            selected_agent.save(checkpoint_path)
        export_env = make_training_env()
        try:
            if (export_env.observation_space.shape, export_env.unwrapped.action_space.shape) in {
                ((3,), (1,)), ((6,), (2,)),
            }:
                policy_path, trajectory_path = export_for_matlab(selected_agent, export_env, output_dir)
                resolved_config["metadata"]["matlab_export"] = {
                    "status": "exported", "policy": str(policy_path.resolve()),
                    "trajectory": str(trajectory_path.resolve()),
                }
            else:
                resolved_config["metadata"]["matlab_export"] = {
                    "status": "unsupported_environment",
                    "reason": "MATLAB export supports altitude (3/1) and altitude/yaw (6/2)",
                }
                print("MATLAB export skipped: unsupported observation/action signature")
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
    curriculum=None,
    from_scratch=False,
    **params,
):
    """Train DQN or SAC, export the selected checkpoint, and return agent/path.

    DQN uses the altitude command grid from its config. SAC keeps the
    environment's continuous actions. Explicit arguments override the config.
    ``eval_freq`` in the config counts total environment transitions.
    SAC uses a fixed mission unless ``curriculum_difficulty`` is configured
    or CurriculumCallback(...) is passed explicitly.
    ``initial_checkpoint`` starts SAC from saved weights and optimizer state.
    The timestep budget includes the checkpoint's existing steps. Its replay
    buffer is not saved in the zip, so collect a fresh warmup before updates.
    Pass ``from_scratch=True`` to ignore the configured starting checkpoint.
    """
    algorithm = algorithm.lower()
    if algorithm not in {"dqn", "sac"}:
        raise ValueError("algorithm must be 'dqn' or 'sac'")
    config_path = CONFIG_DIR / f"{algorithm}.json" if config_path is None else config_path
    config = json.loads(Path(config_path).expanduser().read_text())
    if curriculum is None and config.get("curriculum_difficulty") is not None:
        curriculum = CurriculumCallback(difficulty=config["curriculum_difficulty"], verbose=1)
    if curriculum is not None and (algorithm != "sac" or not issubclass(env_class, DroneEnv)):
        raise ValueError("Curriculum requires SAC with the z-only DroneEnv")
    total_timesteps = config["total_timesteps"] if total_timesteps is None else total_timesteps
    seed = config["seed"] if seed is None else seed
    n_envs = config["n_envs"] if n_envs is None else n_envs
    params = {**config[algorithm], **params}
    initial_checkpoint = None if from_scratch else configured_checkpoint(config, "initial_checkpoint")
    if initial_checkpoint is not None:
        if algorithm != "sac":
            raise ValueError("initial_checkpoint is supported for SAC only")
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(f"Starting checkpoint not found: {initial_checkpoint}")
    eval_freq = config.get("eval_freq", 20_480)
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if not isinstance(eval_freq, int) or isinstance(eval_freq, bool) or eval_freq <= 0:
        raise ValueError("eval_freq must be a positive integer number of transitions")

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

    def make_env(training=False):
        env = env_class(rnn_model=rnn_model, mission=mission, **(
            {"difficulty": curriculum.difficulty} if training and curriculum is not None else {}
        ))
        return env if commands is None else DiscreteDroneActions(env, commands)

    resolved_config = {
        "total_timesteps": total_timesteps,
        "seed": seed,
        "n_envs": n_envs,
        algorithm: params,
        "eval_freq": eval_freq,
        "curriculum_difficulty": None if curriculum is None else curriculum.difficulty,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint is not None else None,
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
    if curriculum is not None:
        resolved_config["curriculum"] = {
            "difficulty": curriculum.difficulty, "heights": list(curriculum.heights),
            "success_threshold": curriculum.success_threshold,
            "window_size": curriculum.successes.maxlen,
        }
    config_snapshot.write_text(json.dumps(resolved_config, indent=2) + "\n")
    train_env = make_vec_env(
        lambda: make_env(training=True), n_envs=n_envs, seed=seed, monitor_dir=str(output_dir),
    )
    eval_env = make_vec_env(make_env, n_envs=1, seed=100)
    agent = None
    agent_class = DQN if algorithm == "dqn" else SAC
    try:
        initial_steps = 0
        if initial_checkpoint is None:
            agent = agent_class(
                "MlpPolicy", train_env, seed=seed, device="cpu", verbose=0,
                tensorboard_log=str(output_dir / "tensorboard"), **params,
            )
        else:
            # SB3 adds use_sde to saved SAC policy_kwargs internally.
            load_params = {**params, "policy_kwargs": {
                **params.get("policy_kwargs", {}), "use_sde": params.get("use_sde", False),
            }}
            agent = agent_class.load(
                initial_checkpoint, env=train_env, seed=seed, device="cpu", verbose=0,
                tensorboard_log=str(output_dir / "tensorboard"), **load_params,
            )
            initial_steps = agent.num_timesteps
            if total_timesteps <= initial_steps:
                raise ValueError(
                    f"total_timesteps ({total_timesteps}) must exceed the starting policy's "
                    f"{initial_steps} steps; use from_scratch=True for a fresh run"
                )
            agent.learning_starts = initial_steps + params["learning_starts"]
            resolved_config["metadata"]["warm_start"] = {
                "checkpoint": str(initial_checkpoint),
                "initial_timesteps": initial_steps,
                "replay_buffer_restored": False,
                "fresh_replay_warmup_transitions": params["learning_starts"],
                "updates_start_after_total_timesteps": agent.learning_starts,
            }
            config_snapshot.write_text(json.dumps(resolved_config, indent=2) + "\n")
        callback_class = _WarmStartEvalCallback if initial_checkpoint is not None else EvalCallback
        callback = callback_class(
            eval_env,
            best_model_save_path=str(output_dir),
            log_path=str(output_dir),
            eval_freq=max(1, eval_freq // n_envs),
            n_eval_episodes=1,
            deterministic=True,
            verbose=0,
        )
        callbacks = callback if curriculum is None else [curriculum, callback]
        agent.learn(
            total_timesteps=total_timesteps - initial_steps, callback=callbacks,
            reset_num_timesteps=initial_checkpoint is None,
            tb_log_name=algorithm.upper(),
        )
        agent.logger.dump(step=agent.num_timesteps)
        final_path = output_dir / f"{algorithm}_drone_rnn.zip"
        agent.save(final_path)
        best_path = output_dir / "best_model.zip"
        # Keep the policy trained through all stages when using a curriculum.
        checkpoint_path = (
            best_path if curriculum is None and best_path.is_file()
            and callback.best_mean_reward > float("-inf") else final_path
        )
        resolved_config["metadata"].update(
            actual_timesteps=agent.num_timesteps,
            new_timesteps=agent.num_timesteps - initial_steps,
            checkpoint=str(checkpoint_path.resolve()),
        )
        resolved_config["checkpoint"] = str(checkpoint_path.resolve())
        if initial_checkpoint is not None:
            resolved_config["metadata"]["warm_start"]["initial_eval_return"] = callback.initial_mean_reward
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
    parser.add_argument("--from-scratch", action="store_true",
                        help="Ignore a configured initial checkpoint (SAC only)")
    parser.add_argument("--curriculum", type=int, nargs="?", const=0, metavar="K",
                        help="Cycle z heights for PPO/SAC, starting at index K (default: 0)")
    parser.add_argument("--seed", type=int, help="Random seed; overrides the config")
    parser.add_argument("--output-dir", type=Path, help="Directory for the training run and results")
    parser.add_argument("--show", action="store_true", help="Also display the saved plots")
    args = parser.parse_args()
    if args.curriculum is not None and args.policy == "dqn":
        parser.error("--curriculum supports PPO or SAC")
    if args.from_scratch and args.policy != "sac":
        parser.error("--from-scratch is only needed for SAC's configured starting policy")
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
    if args.curriculum is not None:
        training_args["curriculum"] = CurriculumCallback(difficulty=args.curriculum, verbose=1)
    if args.policy == "ppo":
        agent, checkpoint_path = train_ppo(rnn_model, ref_mission, training_dir, **training_args)
    else:
        agent, checkpoint_path = train_off_policy(
            args.policy, rnn_model, ref_mission, training_dir,
            from_scratch=args.from_scratch, **training_args,
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

