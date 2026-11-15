"""Altitude control with a learned acceleration model.

The policy observes [z - z_ref, vz, az] and commands the same input to all
four motors. Each environment keeps its own recurrent state. Acceleration
is predicted in normalized units, restored to m/s², then integrated at dt.
"""

from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from model.rnn_loader import checkpoint_parameters

# Statistics used to train the bundled RNN; newer training bundles carry their own.
MEAN_U = -0.004098882433027029
SIGMA_U = 1.7267645597457886
MEAN_AZ = -9.68746566772461
SIGMA_AZ = 21.440523147583008

# Physical parameters of the system
PARAMS = SimpleNamespace(
    dt=0.05,            # Sampling period (s)
    g=9.816,            # Gravitational acceleration (m/s²)
    mean_u=MEAN_U,
    sigma_u=SIGMA_U,
    mean_az=MEAN_AZ,
    sigma_az=SIGMA_AZ,
)

# Landing and operational thresholds
Z_LAND = -0.3          # Landing platform altitude along vertical axis (m)
Z_MAX = 10.0           # Ceiling altitude limit (m)
Z_TRACK_TOLERANCE = 0.2  # Tracking deadband for strictly positive references (m)
SAFE_VZ = 0.5          # Maximum vertical touchdown speed for soft landing (m/s)
MAX_STEPS = 200        # Mission length: 10 s at Ts = 0.05 s
CURRICULUM_HEIGHTS = (5.0, 2.0, 1.0, 0.5)  # Curriculum heights for training (m)


def default_mission(t):
    """Hold at 0 m until 2 s, target 6 m until 6 s, then land."""
    if t < 2.0:
        return 0.0
    if t < 6.0:
        return 6.0
    return Z_LAND


def dynamics(state, u, dt=0.05, rnn_model=None, hidden_state=None, p=PARAMS):
    """Return the next [z, vz, az] state and recurrent memory.

    Motor commands use physical units. The RNN returns normalized
    acceleration; semi-implicit Euler updates velocity before position.
    """
    if rnn_model is None:
        raise ValueError("Altitude dynamics requires a trained RNN")

    z, vz, _ = state
    normalized_u = (float(u) - p.mean_u) / p.sigma_u
    inputs = torch.tensor([[normalized_u]], dtype=torch.float32)
    with torch.no_grad():
        prediction, next_hidden = rnn_model(inputs, hidden_state)
    normalized_az = float(prediction.detach().cpu().numpy().reshape(-1)[0])
    az = normalized_az * p.sigma_az + p.mean_az

    vz_next = vz + az * dt
    z_next = z + vz_next * dt
    return np.array([z_next, vz_next, az], dtype=np.float32), next_hidden


def compute_reward(state, u, outcome, z_ref=0.0, u_prev=0.0):
    """Reward tracking and landing; positive targets have a 0.2 m deadband."""

    def huber_loss(error, delta=1.0):
        if abs(error) <= delta:
            return 0.5 * error**2
        return delta * (abs(error) - 0.5 * delta)

    if outcome == "landed":
        return 1000.0
    if outcome in ("crashed", "lost", "timeout"):
        return -1000.0

    z, vz, _ = state
    # Penalize distance outside the band; zero/negative targets stay exact.
    z_error = z - z_ref
    if z_ref > 0.0:
        z_error = max(abs(z_error) - Z_TRACK_TOLERANCE, 0.0)
    r_track = -10.0 * huber_loss(z_error)
    r_vel = -1.0 * huber_loss(vz)
    r_act = -1.0 * u**2
    # Discourage abrupt changes in motor command.
    r_smooth = -1.0 * abs(u - u_prev)

    return float(r_track + r_vel + r_act + r_smooth)


class DroneEnv(gym.Env):
    """Gymnasium environment for 1D vertical drone flight and landing."""

    metadata = {"render_modes": []}

    def __init__(
        self, rnn_model=None, params=PARAMS, max_steps=MAX_STEPS,
        mission=None, difficulty=None,
    ):
        """Use checkpoint normalization and an optional altitude curriculum."""
        super().__init__()
        self.p = checkpoint_parameters(rnn_model, params, "z") if params is PARAMS else params
        self.max_steps = max_steps
        self.rnn_model = rnn_model

        # Action: single control intensity u in [-3.0, 3.0]
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)

        # Observation: altitude error, vertical velocity, acceleration.
        high = np.array([np.inf, 20.0, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state = None
        self.hidden_state = None
        self.steps = 0
        self.t = 0.0
        self.u_prev = 0.0
        self.mission = mission if mission is not None else default_mission
        if not callable(self.mission):
            raise TypeError("mission must be callable: mission(t) -> target altitude")
        self.difficulty = None
        self._next_difficulty = None
        self._target_reached = False
        if difficulty is not None:
            self.set_difficulty(difficulty)

    def set_difficulty(self, level):
        """Queue a height index for the next reset; the current mission stays fixed."""
        if not isinstance(level, (int, np.integer)):
            raise TypeError("difficulty must be an integer height index")
        self._next_difficulty = int(level) % len(CURRICULUM_HEIGHTS)

    def _obs(self):
        """Use only the reference at the current time; provide no future targets."""
        obs = self.state.copy().astype(np.float32)
        obs[0] -= self.mission(self.t)
        return obs

    def outcome(self):
        """Return flying, landed, crashed, lost, or timeout at the mission deadline."""
        z, vz, _ = self.state
        if z <= Z_LAND:
            soft = abs(vz) <= SAFE_VZ
            landing_phase = self.mission(self.t) == Z_LAND
            return "landed" if soft and landing_phase else "crashed"
        # The 10 m curriculum target needs room for its tracking tolerance.
        ceiling = Z_MAX if self.difficulty is None else max(
            Z_MAX, CURRICULUM_HEIGHTS[self.difficulty] + Z_TRACK_TOLERANCE,
        )
        if z > ceiling:
            return "lost"
        if self.steps >= self.max_steps:
            return "timeout"
        return "flying"

    def reset(self, seed=None, options=None, new_mission=None):
        """Reset state; new_mission replaces the mission until changed again."""
        if new_mission is not None:
            if not callable(new_mission):
                raise TypeError("mission must be callable: mission(t) -> target altitude")
            self.mission = new_mission
            self._next_difficulty = None
            self.difficulty = None
        elif self._next_difficulty is not None:
            self.difficulty = self._next_difficulty
            height = CURRICULUM_HEIGHTS[self.difficulty]
            self.mission = lambda t: 0.0 if t < 2.0 else (height if t < 6.0 else Z_LAND)
        super().reset(seed=seed, options=options)
        z0 = 0.0
        vz0 = 0.0
        az0 = -float(self.p.g)

        self.state = np.array([z0, vz0, az0], dtype=np.float32)
        self.hidden_state = None
        self.steps = 0
        self.t = 0.0
        self.u_prev = 0.0
        self._target_reached = False

        info = {"outcome": "flying"}
        return self._obs(), info

    def step(self, action):
        """Advance dynamics, update mission success, and score the command."""
        u = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -3.0, 3.0))
        old = self.state

        self.state, self.hidden_state = dynamics(
            old,
            u,
            dt=self.p.dt,
            rnn_model=self.rnn_model,
            hidden_state=self.hidden_state,
            p=self.p,
        )

        self.steps += 1
        self.t = self.steps * self.p.dt
        status = self.outcome()

        z_ref = self.mission(self.t)
        if z_ref > 0.0 and abs(float(self.state[0]) - z_ref) <= Z_TRACK_TOLERANCE:
            self._target_reached = True
        reward = compute_reward(self.state, u, status, z_ref=z_ref, u_prev=self.u_prev)

        terminated = (status != "flying")
        truncated = False  # The mission deadline is a terminal failure, not a rollout cutoff.

        info = {
            "outcome": status,
            "difficulty": self.difficulty,
            "is_success": status == "landed" and self._target_reached,
        }
        self.u_prev = u

        return self._obs(), float(reward), terminated, truncated, info
