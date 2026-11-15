"""Altitude and yaw control using rnn_model_thrust_speeds.

Actions [thrust, yaw] become RNN inputs [thrust, roll=0, pitch=0, yaw].
The outputs [az, psi_ddot] are denormalized and integrated without extra physics.
Observations are [z-z_ref, vz, az, psi-psi_ref, psi_dot, psi_ddot].
The existing 18-state layout is kept for saved trajectories; other axes stay zero.
"""

from types import SimpleNamespace
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from model.rnn_loader import checkpoint_parameters

# Training statistics: inputs [thrust, roll, pitch, yaw], outputs [az, psi_ddot].
MEAN_U = np.array([-1.3672e-02,  0.0000e+00,  0.0000e+00,  9.1857e-05], dtype=np.float32)
SIGMA_U = np.array([1.6101, 0.0000, 0.0000, 0.1147], dtype=np.float32)
MEAN_AZ = -10.0084
SIGMA_AZ = 20.6563
MEAN_WZ_DOT = 0.0174
SIGMA_WZ_DOT = 10.8079

# Physical parameters of the system
PARAMS = SimpleNamespace(
    dt=0.05,            # Sampling period (s)
    g=9.816,            # Gravitational acceleration (m/s²)
    mean_u=MEAN_U,
    sigma_u=SIGMA_U,
    mean_az=MEAN_AZ,
    sigma_az=SIGMA_AZ,
    mean_wz_dot=MEAN_WZ_DOT,
    sigma_wz_dot=SIGMA_WZ_DOT,
)

# Running rewards are rates per second; terminal rewards are fixed.


# Landing and operational thresholds
Z_LAND = -0.3          # Landing platform altitude along vertical axis (m)
Z_MAX = 10.0           # Ceiling altitude limit (m)
SAFE_VZ = 0.5          # Maximum vertical touchdown speed for soft landing (m/s)
SAFE_ANGULAR_SPEED = 0.5     # Maximum yaw speed at touchdown (rad/s)
MAX_STEPS = 400        # Mission length: 20 s at Ts = 0.05 s
ALTITUDE_TOLERANCE = 0.2
ALTITUDE_HOLD_TIME = 0.5


def default_mission(t):
    """Climb to 2 m, turn once around z, then land holding the final yaw."""
    if t < 2.0:
        return (0.0, 0.0)
    elif t < 5.0:
        return (2.0, 0.0)
    elif t < 8.0:
        return (2.0, float(2 * np.pi * (t - 5.0) / 3.0))
    else:
        return (Z_LAND, float(2 * np.pi))



# Backward-compatible import for notebooks using the original function name.
mission = default_mission


def dynamics(state, u, dt=0.05, rnn_model=None, hidden_state=None, p=PARAMS, integrate_z=False):
    """Predict [az, psi_ddot] and integrate z and yaw; keep the other axes zero."""
    if rnn_model is None:
        raise ValueError("dynamics requires an rnn_model. Fallback has been disabled.")

    thrust, yaw = np.asarray(u, dtype=np.float32).reshape(2)
    rnn_input = np.array([thrust, 0.0, 0.0, yaw], dtype=np.float32)
    # Keep the supplied statistics; use scale 1 for zero-variance channels.
    scale = np.where(p.sigma_u == 0.0, 1.0, p.sigma_u)
    u_norm = (rnn_input - p.mean_u) / scale
    u_tensor = torch.from_numpy(u_norm).reshape(1, 1, 4)

    with torch.no_grad():
        prediction, next_hidden = rnn_model(u_tensor, hidden_state)
    pred = prediction.detach().cpu().numpy().reshape(-1)
    if pred.size != 2:
        raise ValueError("Expected rnn_model_thrust_speeds outputs [az, psi_ddot]")
    az = float(pred[0]) * p.sigma_az + p.mean_az
    psi_ddot = float(pred[1]) * p.sigma_wz_dot + p.mean_wz_dot

    next_state = np.zeros(18, dtype=np.float32)
    next_state[1] = state[1] + az * dt       # vz
    next_state[0] = state[0] + next_state[1] * dt
    next_state[2] = az
    next_state[16] = state[16] + psi_ddot * dt  # wz = psi_dot
    next_state[11] = state[11] + next_state[16] * dt
    next_state[17] = psi_ddot
    return next_state, next_hidden


REWARD_PARAMS = SimpleNamespace(
    alive_bonus=2.0,
    z_bonus=4.0,
    psi_bonus=4.0,
    z_penalty=2.0,
    psi_penalty=1.0,
    vz_penalty=0.2,
    wz_penalty=0.1,
    yaw_effort=0.01,
    smoothness=0.02,
    success=100.0,
    failure=-25.0,
)


def compute_reward(old_state, state, u, outcome, target_ref=(0.0, 0.0), u_prev=None, dt=PARAMS.dt):
    """Reward tracking and survival over dt, with separate terminal rewards."""
    hubber_loss = lambda x, delta=1.0: 0.5 * x**2 if abs(x) <= delta else delta * (abs(x) - 0.5 * delta)
    r = REWARD_PARAMS

    if isinstance(outcome, tuple):
        z_status, psi_status = outcome
    else:
        z_status, psi_status = outcome, "has_turned"

    if z_status == "landed":
        return r.success if psi_status == "has_turned" else -abs(r.failure)
    if z_status in ("crashed", "lost", "timeout", "missed_altitude"):
        return -abs(r.failure)

    z_ref, psi_ref = target_ref
    z = state[0]
    vz = state[1]
    psi = state[11]
    wz = state[16]

    # Positive tracking bonuses, with smaller penalties away from the targets.
    r_z = r.z_bonus / (1.0 + (z - z_ref)**2) - r.z_penalty * hubber_loss(z - z_ref)

    # Same bonus for unwrapped yaw, keeping 0 and a complete turn distinct.
    r_psi = r.psi_bonus / (1.0 + (psi - psi_ref)**2) - r.psi_penalty * hubber_loss(psi - psi_ref)

    # Velocity penalties
    r_vel = -r.vz_penalty * hubber_loss(vz) - r.wz_penalty * hubber_loss(wz)

    # Penalize yaw effort.
    u_arr = np.asarray(u, dtype=np.float32).reshape(-1)

    # The second element is the direct yaw input.
    r_act = -r.yaw_effort * float(u_arr[1] ** 2)

    # Smoothness penalty
    if u_prev is not None:
        u_prev_arr = np.asarray(u_prev, dtype=np.float32)
        r_smooth = -r.smoothness * float(np.sum(np.abs(u_arr - u_prev_arr)))
    else:
        r_smooth = 0.0


    return float(dt * (r.alive_bonus + r_z + r_psi + r_vel + r_act + r_smooth))


class DroneEnv(gym.Env):
    """Gymnasium environment for altitude, yaw rotation, and landing."""
    metadata = {"render_modes": []}

    out = SimpleNamespace(
        flying="flying",
        landed="landed",
        crashed="crashed",
        lost="lost",
        timeout="timeout",
        has_turned="has_turned",
        no_turn="no_turn"
    )

    def __init__(self, rnn_model=None, params=PARAMS, max_steps=MAX_STEPS, integrate_z=False, mission=None):
        super().__init__()
        self.p = checkpoint_parameters(rnn_model, params, "z_psi") if params is PARAMS else params
        self.max_steps = max_steps
        self.rnn_model = rnn_model
        self.integrate_z = integrate_z
        self.mission = mission if mission is not None else default_mission
        if not callable(self.mission):
            raise TypeError("mission must be callable: mission(t) -> (z_ref, psi_ref)")

        # Direct control inputs [thrust, yaw] in [-3.0, 3.0]^2
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(2,), dtype=np.float32)

        # [z-z_ref, vz, az, psi-psi_ref, psi_dot, psi_ddot]
        high = np.array([Z_MAX, 20.0, 100.0, np.inf, np.inf, 500.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state = None
        self.hidden_state = None
        self.steps = 0
        self.t = 0.0
        self.u_prev = np.zeros(2, dtype=np.float32)
        self.az_prev = -self.p.g
        self.has_turned = False
        self.altitude_hold_time = 0.0
        self.out = self.__class__.out

    def _obs(self):
        """Return the altitude and yaw tracking observation."""
        z_ref, psi_ref = self.mission(self.t)
        return np.array([
            self.state[0] - z_ref,
            self.state[1],
            self.state[2],
            self.state[11] - psi_ref,
            self.state[16],
            self.state[17],
        ], dtype=np.float32)

    def outcome(self):
        """Return (z status, yaw status) for the selected mission's final yaw."""
        z = self.state[0]
        psi = self.state[11]
        vz = self.state[1]
        z_ref, _ = self.mission(self.t)

        _, psi_goal = self.mission(self.max_steps * self.p.dt)
        # Unwrapped error supports positive/negative turns and climb-only missions.
        self.has_turned = bool(abs(psi - psi_goal) <= np.deg2rad(10.0))
        psi_status = self.out.has_turned if self.has_turned else self.out.no_turn
        if z <= Z_LAND:
            soft = abs(vz) <= SAFE_VZ
            settled = abs(self.state[16]) <= SAFE_ANGULAR_SPEED
            landing_phase = (z_ref == Z_LAND)
            z_status = self.out.landed if soft and settled and landing_phase else self.out.crashed
            if z_status == self.out.landed and self.altitude_hold_time < ALTITUDE_HOLD_TIME:
                z_status = "missed_altitude"
        elif z > Z_MAX:
            z_status = self.out.lost
        elif self.steps >= self.max_steps:
            z_status = self.out.timeout
        else:
            z_status = self.out.flying
        return z_status, psi_status

    def reset(self, seed=None, options=None, new_mission=None):
        """Resets the environment to initial state; new_mission replaces the active mission."""
        if new_mission is not None:
            if not callable(new_mission):
                raise TypeError("mission must be callable: mission(t) -> (z_ref, psi_ref)")
            self.mission = new_mission
        super().reset(seed=seed, options=options)

        self.state = np.zeros(18, dtype=np.float32)
        self.state[0] = 0.0              # z
        self.state[1] = 0.0              # vz
        self.state[2] = -float(self.p.g) # az

        self.hidden_state = None
        self.steps = 0
        self.t = 0.0
        self.u_prev = np.zeros(2, dtype=np.float32)
        self.az_prev = self.state[2]
        self.has_turned = False

        requires_climb = any(self.mission(i * self.p.dt)[0] > 0.0
                             for i in range(self.max_steps + 1))
        self.altitude_hold_time = 0.0 if requires_climb else ALTITUDE_HOLD_TIME

        z_ref, psi_ref = self.mission(self.t)
        info = {
            "outcome": self.outcome(),
            "m": {
                "t": self.t,
                "z": float(self.state[0]),
                "vz": float(self.state[1]),
                "az_prev": float(self.az_prev),
                "z_ref": float(z_ref),
                "psi_ref": float(psi_ref),
                "u_prev": float(self.u_prev[0]),
                "psi": float(self.state[11]),
            }
        }
        return self._obs(), info

    def step(self, action):
        """Performs one environment step under control input u."""
        u = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -3.0, 3.0)
        if len(u) < 2:
            u = np.pad(u, (0, 2 - len(u)), mode='constant')
        old = self.state

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
        z_ref, _ = self.mission(self.t)
        if z_ref > 0.0 and abs(self.state[0] - z_ref) <= ALTITUDE_TOLERANCE:
            self.altitude_hold_time += self.p.dt
        status = self.outcome()

        target_ref = self.mission(self.t)
        reward = compute_reward(old, self.state, u, status, target_ref=target_ref, u_prev=self.u_prev, dt=self.p.dt)
        z_ref, psi_ref = target_ref

        terminated = (status[0] != self.out.flying)
        truncated = False  # The mission deadline is a terminal failure, not a rollout cutoff.

        info = {
            "outcome": status,
            "is_success": status == (self.out.landed, self.out.has_turned),
            "altitude_hold_time": self.altitude_hold_time,
            "m": {
                "t": self.t,
                "z": float(self.state[0]),
                "vz": float(self.state[1]),
                "az_prev": float(self.az_prev),
                "z_ref": float(z_ref),
                "psi_ref": float(psi_ref),
                "u_prev": float(self.u_prev[0]),
                "psi": float(self.state[11]),
            }
        }

        self.az_prev = float(self.state[2])
        self.u_prev = u.copy()

        return self._obs(), float(reward), terminated, truncated, info


# ── Testing and Evaluation Helpers ──────────────────────────────────────────

def run_episode(env, policy, seed=None, max_steps=None):
    """Return states, actions, return, and outcome for policy(obs) -> action.

    The active mission is available as env.unwrapped.mission, including when
    env is wrapped. By default, use the environment's own episode horizon.
    """
    obs, info = env.reset(seed=seed)
    base = env.unwrapped
    states, actions, total_reward = [base.state.copy()], [], 0.0

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
        is_landed = (outcome == ("landed", "has_turned") if isinstance(outcome, tuple) else outcome == "landed")
        landed += int(is_landed)
    returns = np.array(returns)
    print(f"mean return: {returns.mean():7.1f} ± {returns.std():5.1f} | landed: {landed}/{n}")
    return returns.mean(), landed / n
