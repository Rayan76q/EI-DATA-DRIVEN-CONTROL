"""Generalized 3D Drone Environment for Gymnasium (Track B / Advanced Tasks).

Full 6-DOF rigid body quadrotor drone:
    - 4 control inputs: u = [u1, u2, u3, u4] in [-3.0, 3.0]^4 (motor tensions)
    - 18 state variables:
        [0:3]   z, vz, az          (vertical axis translation)
        [3:6]   y, vy, ay          (lateral axis translation)
        [6:9]   x, vx, ax          (longitudinal axis translation)
        [9:12]  theta, phi, psi    (Euler angles: pitch, roll, yaw)
        [12:14] wx, wx_dot         (angular velocity and acceleration around roll axis)
        [14:16] wy, wy_dot         (angular velocity and acceleration around pitch axis)
        [16:18] wz, wz_dot         (angular velocity and acceleration around yaw axis)

    - RNN dynamics: predicts 8 double derivatives / angles:
        [ax, ay, az, wx_dot, wy_dot, wz_dot, phi, psi]
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

# General 3D normalization parameters (placeholders or identified values)
MEAN_AX = 0.0
SIGMA_AX = 10.0
MEAN_AY = 0.0
SIGMA_AY = 10.0
MEAN_THETA = 0.0
SIGMA_THETA = 0.5
MEAN_PHI = 0.0
SIGMA_PHI = 0.5
MEAN_PSI = 0.0
SIGMA_PSI = 1.0
MEAN_WX = 0.0
SIGMA_WX = 5.0
MEAN_WY = 0.0
SIGMA_WY = 5.0
MEAN_WZ = 0.0
SIGMA_WZ = 5.0
MEAN_WX_DOT = 0.0
SIGMA_WX_DOT = 20.0
MEAN_WY_DOT = 0.0
SIGMA_WY_DOT = 20.0
MEAN_WZ_DOT = 0.0
SIGMA_WZ_DOT = 20.0

# Physical parameters of the system
PARAMS = SimpleNamespace(
    dt=0.05,            # Sampling period Ts (s)
    g=9.816,            # Gravitational acceleration (m/s²)
    mean_u=MEAN_U,
    sigma_u=SIGMA_U,
    mean_ax=MEAN_AX,
    sigma_ax=SIGMA_AX,
    mean_ay=MEAN_AY,
    sigma_ay=SIGMA_AY,
    mean_az=MEAN_AZ,
    sigma_az=SIGMA_AZ,
    mean_phi=MEAN_PHI,
    sigma_phi=SIGMA_PHI,
    mean_theta=MEAN_THETA,
    sigma_theta=SIGMA_THETA,
    mean_psi=MEAN_PSI,
    sigma_psi=SIGMA_PSI,
    mean_wx=MEAN_WX,
    sigma_wx=SIGMA_WX,
    mean_wy=MEAN_WY,
    sigma_wy=SIGMA_WY,
    mean_wz=MEAN_WZ,
    sigma_wz=SIGMA_WZ,
    mean_wx_dot=MEAN_WX_DOT,
    sigma_wx_dot=SIGMA_WX_DOT,
    mean_wy_dot=MEAN_WY_DOT,
    sigma_wy_dot=SIGMA_WY_DOT,
    mean_wz_dot=MEAN_WZ_DOT,
    sigma_wz_dot=SIGMA_WZ_DOT,
)

# Operational limits
Z_LAND = -0.3          # Landing platform altitude (m)
Z_MAX = 10.0           # Ceiling limit (m)
XY_MAX = 10.0          # Horizontal workspace limit (m)
SAFE_VZ = 0.5          # Maximum touchdown vertical speed (m/s)
SAFE_VXY = 0.5         # Maximum touchdown horizontal speed (m/s)
MAX_STEPS = 200        # Mission length: 10 s at Ts = 0.05 s


def default_mission(t):
    """Target reference (z_ref, y_ref, x_ref, psi_ref) at time t (seconds)."""
    if t < 2.0:
        return (0.0, 0.0, 0.0, 0.0)
    elif t < 4.0:
        return (0.5, 0.0, 0.0, 0.0)
    elif t < 5.0:
        return (0.5, 0.0, 0.0, float(np.pi / 2.0))
    elif t < 6.0:
        return (0.5, 0.0, 0.0, float(np.pi))
    elif t < 7.0:
        return (0.5, 0.0, 0.0, float(3.0 * np.pi / 2.0))
    elif t < 8.0:
        return (0.5, 0.0, 0.0, float(2.0 * np.pi))
    else:
        return (Z_LAND, 0.0, 0.0, 0.0)


# Backward-compatible import for notebooks using the original function name.
mission = default_mission


def dynamics(state, u, dt=0.05, rnn_model=None, hidden_state=None, p=PARAMS, integrate_z=False):
    """Calculates next 18-variable state and next recurrent hidden state in 3D.

    Args:
        state (np.ndarray): Current 18-element state vector.
        u (np.ndarray): 4-rotor control vector [u1, u2, u3, u4].
        dt (float): Integration step (s).
        rnn_model (optional): Trained PyTorch model predicting 8 variables.
        hidden_state (optional): Recurrent hidden state.
        p (SimpleNamespace): Parameters.
        integrate_z (bool): Unused when predicting az directly.

    Returns:
        tuple: (next_state np.ndarray of shape (18,), next_hidden_state)
    """
    z, vz, az = state[0], state[1], state[2]
    y, vy, ay = state[3], state[4], state[5]
    x, vx, ax = state[6], state[7], state[8]
    theta, phi, psi = state[9], state[10], state[11]
    wx, wx_dot = state[12], state[13]
    wy, wy_dot = state[14], state[15]
    wz, wz_dot = state[16], state[17]

    u_arr = np.asarray(u, dtype=np.float32).reshape(-1)
    if len(u_arr) < 4:
        u_arr = np.pad(u_arr, (0, 4 - len(u_arr)), mode='edge')

    if rnn_model is not None:
        # Normalize 4 motor commands before RNN
        u_norm = (u_arr[:4] - p.mean_u) / p.sigma_u
        u_tensor = torch.tensor([[u_norm.tolist()]], dtype=torch.float32)

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
        # Expected outputs: [ax, ay, az, wx_dot, wy_dot, wz_dot, phi, psi]
        ax_phys = float(pred[0]) * p.sigma_ax + p.mean_ax if len(pred) > 0 else 0.0
        ay_phys = float(pred[1]) * p.sigma_ay + p.mean_ay if len(pred) > 1 else 0.0
        az_phys = float(pred[2]) * p.sigma_az + p.mean_az if len(pred) > 2 else -p.g
        wx_dot_phys = float(pred[3]) * p.sigma_wx_dot + p.mean_wx_dot if len(pred) > 3 else 0.0
        wy_dot_phys = float(pred[4]) * p.sigma_wy_dot + p.mean_wy_dot if len(pred) > 4 else 0.0
        wz_dot_phys = float(pred[5]) * p.sigma_wz_dot + p.mean_wz_dot if len(pred) > 5 else 0.0
        phi_phys = float(pred[6]) * p.sigma_phi + p.mean_phi if len(pred) > 6 else 0.0
        psi_phys = float(pred[7]) * p.sigma_psi + p.mean_psi if len(pred) > 7 else 0.0
    else:
        next_hidden = hidden_state
        # Analytical 4-rotor baseline model
        u1, u2, u3, u4 = u_arr[0], u_arr[1], u_arr[2], u_arr[3]
        u_mean = float(0.25 * (u1 + u2 + u3 + u4))
        u_phys = u_mean * p.sigma_u + p.mean_u

        # Vertical acceleration
        az_phys = float(18.3424 * u_phys - p.g)

        # Angular accelerations from motor differences
        # Roll torque: u4 - u2; Pitch torque: u3 - u1; Yaw torque: (u1 + u3) - (u2 + u4)
        wx_dot_phys = float(5.0 * (u4 - u2))
        wy_dot_phys = float(5.0 * (u3 - u1))
        wz_dot_phys = float(2.0 * ((u1 + u3) - (u2 + u4)))

        # Euler angle integration from angular rates
        phi_phys = float(phi + wx * dt)
        psi_phys = float(psi + wz * dt)
        ax_phys = float(p.g * theta)
        ay_phys = float(-p.g * phi)

    # 1. Linear kinematics integration
    vz_next = vz + az_phys * dt
    z_next = z + vz_next * dt

    vy_next = vy + ay_phys * dt
    y_next = y + vy_next * dt

    vx_next = vx + ax_phys * dt
    x_next = x + vx_next * dt

    # 2. Angular rates integration
    wx_next = wx + wx_dot_phys * dt
    wy_next = wy + wy_dot_phys * dt
    wz_next = wz + wz_dot_phys * dt

    # 3. Attitude propagation
    phi_next = phi_phys
    psi_next = psi_phys
    theta_next = float(theta + wy_next * dt)

    next_state = np.array([
        z_next, vz_next, az_phys,
        y_next, vy_next, ay_phys,
        x_next, vx_next, ax_phys,
        theta_next, phi_next, psi_next,
        wx_next, wx_dot_phys,
        wy_next, wy_dot_phys,
        wz_next, wz_dot_phys
    ], dtype=np.float32)

    return next_state, next_hidden


def compute_reward(old_state, state, u, outcome, target_ref=(0.0, 0.0, 0.0, 0.0), u_prev=None):
    """Computes full 3D tracking error and smoothness reward."""
    hubber_loss = lambda val, delta=1.0: 0.5 * val**2 if abs(val) <= delta else delta * (abs(val) - 0.5 * delta)

    if outcome == "landed":
        return 1000.0
    if outcome in ("crashed", "lost", "timeout"):
        return -1000.0

    if len(target_ref) == 4:
        z_ref, y_ref, x_ref, psi_ref = target_ref
    elif len(target_ref) == 2:
        z_ref, psi_ref = target_ref
        y_ref, x_ref = 0.0, 0.0
    else:
        z_ref, y_ref, x_ref, psi_ref = 0.0, 0.0, 0.0, 0.0

    z, vz = state[0], state[1]
    y, vy = state[3], state[4]
    x, vx = state[6], state[7]
    theta, phi, psi = state[9], state[10], state[11]
    wx, wy, wz = state[12], state[14], state[16]

    # Position tracking penalties
    r_pos = -10.0 * hubber_loss(z - z_ref) - 5.0 * hubber_loss(y - y_ref) - 5.0 * hubber_loss(x - x_ref)

    # Orientation tracking (psi wrapped in [-pi, pi])
    psi_err = (psi - psi_ref + np.pi) % (2.0 * np.pi) - np.pi
    r_att = -10.0 * hubber_loss(psi_err) - 2.0 * (abs(theta) + abs(phi))

    # Velocity penalties
    r_vel = -1.0 * hubber_loss(vz) - 0.5 * (hubber_loss(vx) + hubber_loss(vy))
    r_rates = -0.2 * (wx**2 + wy**2 + wz**2)

    # Control effort
    u_arr = np.asarray(u, dtype=np.float32)
    r_act = -1.0 * float(np.sum(u_arr ** 2))

    # Command smoothness
    if u_prev is not None:
        u_prev_arr = np.asarray(u_prev, dtype=np.float32)
        r_smooth = -1.0 * float(np.sum(np.abs(u_arr - u_prev_arr)))
    else:
        r_smooth = 0.0

    return float(r_pos + r_att + r_vel + r_rates + r_act + r_smooth)


class DroneEnv(gym.Env):
    """Generalized Gymnasium environment for full 3D quadcopter drone flight."""
    metadata = {"render_modes": []}

    def __init__(self, rnn_model=None, params=PARAMS, max_steps=MAX_STEPS, integrate_z=False, mission=None):
        super().__init__()
        self.p = params
        self.max_steps = max_steps
        self.rnn_model = rnn_model
        self.integrate_z = integrate_z
        self.mission = mission if mission is not None else default_mission
        if not callable(self.mission):
            raise TypeError("mission must be callable: mission(t) -> target")

        # Generalized action space: 4 motor commands in [-3.0, 3.0]
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(4,), dtype=np.float32)

        # Observation space: 18 continuous state variables
        high = np.array([
            Z_MAX, 20.0, 100.0,        # z, vz, az
            XY_MAX, 20.0, 100.0,       # y, vy, ay
            XY_MAX, 20.0, 100.0,       # x, vx, ax
            np.pi, np.pi, 2.0 * np.pi, # theta, phi, psi
            50.0, 500.0,               # wx, wx_dot
            50.0, 500.0,               # wy, wy_dot
            50.0, 500.0                # wz, wz_dot
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state = None
        self.hidden_state = None
        self.steps = 0
        self.t = 0.0
        self.u_prev = np.zeros(4, dtype=np.float32)
        self.az_prev = -self.p.g

    def _obs(self):
        """Returns 18-element state tracking error vector."""
        obs = self.state.copy().astype(np.float32)
        ref = self.mission(self.t)
        if len(ref) == 4:
            z_ref, y_ref, x_ref, psi_ref = ref
        elif len(ref) == 2:
            z_ref, psi_ref = ref
            y_ref, x_ref = 0.0, 0.0
        else:
            z_ref, y_ref, x_ref, psi_ref = 0.0, 0.0, 0.0, 0.0

        obs[0] -= z_ref
        obs[3] -= y_ref
        obs[6] -= x_ref
        obs[11] -= psi_ref
        return obs

    def outcome(self):
        """Flight status: 'flying', 'landed', 'crashed', 'lost', or 'timeout'."""
        z, vz = self.state[0], self.state[1]
        y, x = self.state[3], self.state[6]
        vy, vx = self.state[4], self.state[7]

        ref = self.mission(self.t)
        z_ref = ref[0]

        if z <= Z_LAND:
            soft = (abs(vz) <= SAFE_VZ) and (np.hypot(vx, vy) <= SAFE_VXY)
            landing_phase = (z_ref == Z_LAND)
            return "landed" if soft and landing_phase else "crashed"

        if z > Z_MAX or abs(x) > XY_MAX or abs(y) > XY_MAX:
            return "lost"

        if self.steps >= self.max_steps:
            return "timeout"

        return "flying"

    def reset(self, seed=None, options=None, new_mission=None):
        """Resets the 3D environment to origin; new_mission replaces the active mission."""
        if new_mission is not None:
            if not callable(new_mission):
                raise TypeError("mission must be callable: mission(t) -> target")
            self.mission = new_mission
        super().reset(seed=seed, options=options)

        self.state = np.zeros(18, dtype=np.float32)
        self.state[0] = 0.0              # z
        self.state[1] = 0.0              # vz
        self.state[2] = -float(self.p.g) # az

        self.hidden_state = None
        self.steps = 0
        self.t = 0.0
        self.u_prev = np.zeros(4, dtype=np.float32)
        self.az_prev = self.state[2]

        ref = self.mission(self.t)
        z_ref = ref[0]
        psi_ref = ref[3] if len(ref) >= 4 else (ref[1] if len(ref) == 2 else 0.0)

        info = {
            "outcome": "flying",
            "m": {
                "t": self.t,
                "z": float(self.state[0]),
                "vz": float(self.state[1]),
                "az_prev": float(self.az_prev),
                "z_ref": float(z_ref),
                "u_prev": float(self.u_prev[0]),
                "psi": float(self.state[11]),
                "psi_ref": float(psi_ref),
            }
        }
        return self._obs(), info

    def step(self, action):
        """Executes one simulation step under 4 rotor controls."""
        u = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -3.0, 3.0)
        if len(u) < 4:
            u = np.pad(u, (0, 4 - len(u)), mode='edge')
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
        status = self.outcome()

        ref = self.mission(self.t)
        reward = compute_reward(old, self.state, u, status, target_ref=ref, u_prev=self.u_prev)

        terminated = (status != "flying")
        truncated = False  # The mission deadline is a terminal failure, not a rollout cutoff.

        z_ref = ref[0]
        psi_ref = ref[3] if len(ref) >= 4 else (ref[1] if len(ref) == 2 else 0.0)

        info = {
            "outcome": status,
            "m": {
                "t": self.t,
                "z": float(self.state[0]),
                "vz": float(self.state[1]),
                "az_prev": float(self.az_prev),
                "z_ref": float(z_ref),
                "u_prev": float(self.u_prev[0]),
                "psi": float(self.state[11]),
                "psi_ref": float(psi_ref),
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
    """Evaluates policy over n episodes."""
    returns, landed = [], 0
    for i in range(n):
        _, _, total, outcome = run_episode(env, policy, seed=seed + i)
        returns.append(total)
        landed += (outcome == "landed")
    returns = np.array(returns)
    print(f"mean return: {returns.mean():7.1f} ± {returns.std():5.1f} | landed: {landed}/{n}")
    return returns.mean(), landed / n
