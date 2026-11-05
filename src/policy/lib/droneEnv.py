"""Drone flying, as a Gymnasium environment (TD 4).

The drone is a rigid body in a 3D space. State (3 numbers):

    z       position of the centre of mass (m)
    vz      velocity along vertical axis (m/s)
    az      acceleration along vertical axis (m/s²)

The policy observes [z - env.mission(t), vz, az].
The environment generates the current reference; the policy receives no mission
function, schedule, future targets, or time-to-switch.

We control all four rotors with the same intensity named u, the movement remains along the vertical axis (Oz).
The platform to land on is at (x,y,z) = (0, 0, -0.3) (not on the car to begin with)


    from policy.lib.droneEnv import DroneEnv
    env = DroneEnv(mission=lambda t: 2.0 if t < 6.0 else -0.3)

"""

from types import SimpleNamespace
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

# Normalization parameters (Z-score statistics from dataset)
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
Z_MAX = 10.0            # Ceiling altitude limit (m)
Z_TRACK_TOLERANCE = 0.2 # Tracking deadband for strictly positive references (m)
SAFE_VZ = 0.5          # Maximum vertical touchdown speed for soft landing (m/s)
MAX_STEPS = 200        # Mission length: 10 s at Ts = 0.05 s
CURRICULUM_HEIGHTS = (0.5, 1.0, 2.0, 5.0)


def default_mission(t):
    """Hold at 0 m until 2 s, target 6 m until 6 s, then land."""
    if t < 2.0:
        return 0.0       # Step 1: On the platform (z = 0)
    elif t < 6.0:
        return 6.0       # Step 2: Hover above to gather data (z = 6)
    else:
        return Z_LAND      # Step 3: Descend and land on the platform (z = -0.3)


# Backward-compatible import for notebooks using the original function name.
mission = default_mission


def dynamics(state, u, dt=0.05, rnn_model=None, hidden_state=None, p=PARAMS, integrate_z=False):
    """Calculates next state [z, vz, az] and next recurrent hidden state.

    The input u is a raw motor command in [-3, 3], standardized once here.
    When rnn_model is provided, it predicts normalized az, which is then
    denormalized to physical units (m/s²).

    Args:
        state (np.ndarray): Current state [z, vz, az].
        u (float): Raw motor command in [-3, 3].
        dt (float): Integration time step in seconds.
        rnn_model (optional): Trained PyTorch RNN/LSTM/GRU model predicting normalized az.
        hidden_state (optional): Previous hidden state tensor of the RNN.
        p (SimpleNamespace): Physical and normalization parameters.
        integrate_z (bool): Unused when predicting az only.

    Returns:
        tuple: (next_state np.ndarray, next_hidden_state)
    """
    z, vz, az = state

    if rnn_model is not None:
        u_tensor = torch.tensor([[(float(u) - p.mean_u)/p.sigma_u ]], dtype=torch.float32)
        with torch.no_grad():
            if hasattr(rnn_model, 'rnn') and hasattr(rnn_model, 'head'):
                out, next_hidden = rnn_model.rnn(u_tensor, hidden_state)
                pred_tensor = rnn_model.head(out)
            else:
                res = rnn_model(u_tensor, hidden_state)
                if isinstance(res, tuple):
                    pred_tensor, next_hidden = res
                else:
                    pred_tensor = res
                    next_hidden = hidden_state

            if hasattr(pred_tensor, 'detach'):
                pred = pred_tensor.detach().cpu().squeeze().numpy()
            else:
                pred = np.asarray(pred_tensor).squeeze()

        pred = np.atleast_1d(pred).astype(float)
        pred_az_norm = float(pred[0])

        # Dénormalisation de az : valeur physique en m/s²
        az_phys = pred_az_norm * p.sigma_az + p.mean_az
    else:
        next_hidden = hidden_state
        # Baseline sans RNN : la commande u est déjà dans ses unités brutes.
        u_phys = float(u)
        az_phys = float(18.3424 * u_phys - p.g)

    # Intégration cinématique physique
    vz_next = vz + az_phys * dt
    z_next = z + vz_next * dt
    az_next = az_phys

    next_state = np.array([z_next, vz_next, az_next], dtype=np.float32)
    return next_state, next_hidden


def compute_reward(old_state, state, u, outcome, z_ref=0.0, u_prev=0.0):
    """Reward tracking and landing; positive targets have a 0.2 m deadband."""

    hubber_loss = lambda x, delta=1.0: 0.5 * x**2 if abs(x) <= delta else delta * (abs(x) - 0.5 * delta)

    if outcome == "landed":
        return 1000.0
    if outcome in ("crashed", "lost", "timeout"):
        return -1000.0

    z, vz, _ = state
    # Penalize distance outside the band; zero/negative targets stay exact.
    z_error = z - z_ref
    if z_ref > 0.0:
        z_error = max(abs(z_error) - Z_TRACK_TOLERANCE, 0.0)
    r_track = -10.0 * hubber_loss(z_error)
    # Velocity penalty
    r_vel = -1.0 * hubber_loss(vz)
    # Control effort penalty
    r_act = -1.0 * u**2
    # Reward staying airborne and discourage abrupt command changes.
    r_smooth = -1.0 * abs(u - u_prev)

    return float(r_track + r_vel + r_act + r_smooth  )


class DroneEnv(gym.Env):
    """Gymnasium environment for 1D vertical drone flight and landing."""
    metadata = {"render_modes": []}

    def __init__(self, rnn_model=None, params=PARAMS, max_steps=MAX_STEPS, integrate_z=False, mission=None, difficulty=None):
        super().__init__()
        self.p = params
        self.max_steps = max_steps
        self.rnn_model = rnn_model
        self.integrate_z = integrate_z

        # Action: single control intensity u in [-3.0, 3.0]
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)

        # Observation: altitude error, vertical velocity, acceleration.
        high = np.array([np.inf, 20.0, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.state = None
        self.hidden_state = None
        self.steps = 0
        self.t = 0.0
        self.u_prev = 0.0
        self.az_prev = -self.p.g
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
        self.az_prev = az0
        self._target_reached = False

        z_ref = self.mission(self.t)
        info = {
            "outcome": "flying",
            "m": {
                "t": self.t,
                "z": z0,
                "vz": vz0,
                "az_prev": az0,
                "z_ref": z_ref,
                "u_prev": self.u_prev
            }
        }
        return self._obs(), info

    def step(self, action):
        """Performs one environment step under control input u."""
        u = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -3.0, 3.0))
        old = self.state

        # Update physical state and internal model memory
        self.state, self.hidden_state = dynamics(
            old,
            u,
            dt=self.p.dt,
            rnn_model=self.rnn_model,
            hidden_state=self.hidden_state,
            p=self.p,
            integrate_z=self.integrate_z
        )

        self.steps += 1
        self.t = self.steps * self.p.dt
        status = self.outcome()

        z_ref = self.mission(self.t)
        if z_ref > 0.0 and abs(float(self.state[0]) - z_ref) <= Z_TRACK_TOLERANCE:
            self._target_reached = True
        reward = compute_reward(old, self.state, u, status, z_ref=z_ref, u_prev=self.u_prev)

        terminated = (status != "flying")
        truncated = False  # The mission deadline is a terminal failure, not a rollout cutoff.

        # Measurements dictionary required by export_policy.py and Simulink evaluation
        info = {
            "outcome": status,
            "difficulty": self.difficulty,
            "is_success": status == "landed" and self._target_reached,
            "m": {
                "t": self.t,
                "z": float(self.state[0]),
                "vz": float(self.state[1]),
                "az_prev": float(self.state[2]),  # Same available acceleration as _obs().
                "z_ref": float(z_ref),
                "u_prev": float(self.u_prev)
            }
        }

        # Update previous step values for next iteration
        self.az_prev = float(self.state[2])
        self.u_prev = u

        return self._obs(), float(reward), terminated, truncated, info


# ── Testing and Evaluation Helpers ──────────────────────────────────────────

def run_episode(env, policy, seed=None, max_steps=None):
    """Return states, actions, return, and outcome for policy(obs) -> action.

    The active mission is available as env.unwrapped.mission, including when
    env is wrapped. By default, use the environment's own episode horizon.
    """
    obs, _ = env.reset(seed=seed)
    base = env.unwrapped
    states, actions, total_reward = [base.state.copy()], [], 0.0
    info = {"outcome": "flying"}

    for _ in range(base.max_steps if max_steps is None else max_steps):
        a = policy(obs)
        obs, r, terminated, truncated, info = env.step(a)
        states.append(base.state.copy())
        actions.append(a)
        total_reward += r
        if terminated or truncated:
            break

    return np.array(states), actions, total_reward, info["outcome"]


def evaluate(env, policy, n=50, seed=0):
    """Evaluates mean return and landing success rate over n episodes."""
    returns, landed = [], 0
    for i in range(n):
        _, _, total, outcome = run_episode(env, policy, seed=seed + i)
        returns.append(total)
        landed += (outcome == "landed")
    returns = np.array(returns)
    print(f"mean return: {returns.mean():7.1f} ± {returns.std():5.1f} | landed: {landed}/{n}")
    return returns.mean(), landed / n

