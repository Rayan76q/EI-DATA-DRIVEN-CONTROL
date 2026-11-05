"""Discrete collective motor commands for DQN."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DiscreteDroneActions(gym.ActionWrapper):
    """Map a DQN action index to one raw collective motor command."""

    def __init__(self, env, commands):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box) or env.action_space.shape != (1,):
            raise ValueError("DiscreteDroneActions requires a scalar Box action space")
        commands = np.asarray(commands, dtype=np.float32)
        if commands.ndim == 2 and commands.shape[1] == 1:
            commands = commands[:, 0]
        if commands.ndim != 1 or commands.size < 2:
            raise ValueError("commands must contain at least two scalar motor commands")
        if not np.all(np.isfinite(commands)) or np.unique(commands).size != commands.size:
            raise ValueError("commands must contain finite, distinct values")
        if np.any(commands < env.action_space.low[0]) or np.any(commands > env.action_space.high[0]):
            raise ValueError("commands must lie within the environment's action bounds")
        self.commands = commands.copy()
        self.action_space = spaces.Discrete(len(self.commands))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid discrete motor command index: {action}")
        return np.array([self.commands[int(action)]], dtype=np.float32)
