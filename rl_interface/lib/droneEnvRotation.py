"""Drone flying with rotation, as a Gymnasium environment (TD 4 / Advanced Tasks).

The drone is a rigid body in 3D space with 18 state variables:
    z, vz, az          (vertical axis translation)
    y, vy, ay          (lateral axis translation)
    x, vx, ax          (longitudinal axis translation)
    theta, phi, psi    (Euler angles: pitch, roll, yaw)
    wx, wx_dot         (angular velocity and acceleration around roll axis)
    wy, wy_dot         (angular velocity and acceleration around pitch axis)
    wz, wz_dot         (angular velocity and acceleration around yaw axis)

Intermediate task:
    - Vertical motion along z and yaw rotation around psi only.
    - x, y, theta, phi remain zero.
    - Action space restricted to 2 rotor commands [u1, u2] in [-3.0, 3.0].
    - RNN currently predicts [az, psi] from [u1, u2].
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

# General case placeholders (to be filled when identified from full dataset)
MEAN_AY = None
SIGMA_AY = None
MEAN_AX = None
SIGMA_AX = None
MEAN_THETA = None
SIGMA_THETA = None
MEAN_PHI = None
SIGMA_PHI = None
MEAN_PSI = 0.0
SIGMA_PSI = 1.0
MEAN_WX = None
SIGMA_WX = None
MEAN_WY = None
SIGMA_WY = None
MEAN_WZ = None
SIGMA_WZ = None
MEAN_WX_DOT = None
SIGMA_WX_DOT = None
MEAN_WY_DOT = None
SIGMA_WY_DOT = None
MEAN_WZ_DOT = None
SIGMA_WZ_DOT = None

# Physical parameters of the system
PARAMS = SimpleNamespace(
    dt=0.05,            # Sampling period (s)
    g=9.816,            # Gravitational acceleration (m/s²)
    mean_u=MEAN_U,
    sigma_u=SIGMA_U,
    mean_az=MEAN_AZ,
    sigma_az=SIGMA_AZ,
    mean_psi=MEAN_PSI,
    sigma_psi=SIGMA_PSI,
)

# Landing and operational thresholds
Z_LAND = -0.3          # Landing platform altitude along vertical axis (m)
Z_MAX = 10.0           # Ceiling altitude limit (m)
SAFE_VZ = 0.5          # Maximum vertical touchdown speed for soft landing (m/s)
SAFE_VX = 0.5
SAFE_VY = 0.5
MAX_STEPS = 200        # Mission length: 10 s at Ts = 0.05 s


def default_mission(t):
    """Target reference altitude z_ref and target yaw angle psi_ref at time t (seconds)."""
    if t < 2.0:
        return (0.0, 0.0)                 # Step 1: On the platform (z = 0)
    elif t < 4.0:
        return (0.5, 0.0)                 # Step 2: Climb to altitude (z = 0.5)
    elif t < 5.0:
        return (0.5, float(np.pi / 2.0))  # Rotate 90 degrees
    elif t < 6.0:
        return (0.5, float(np.pi))        # Rotate 180 degrees
    elif t < 7.0:
        return (0.5, float(3.0 * np.pi / 2.0)) # Rotate 270 degrees
    elif t < 8.0:
        return (0.5, float(2.0 * np.pi))  # Complete rotation (360 degrees)
    else:
        return (Z_LAND, 0.0)              # Step 3: Descend and land on platform (z = -0.3)


# Backward-compatible import for notebooks using the original function name.
mission = default_mission


def dynamics(state, u, dt=0.05, rnn_model=None, hidden_state=None, p=PARAMS, integrate_z=False):
    """Calculates next 18-variable state and next recurrent hidden state.

    Intermediate case (active):
        Inputs: u = [u1, u2] in [-3, 3].
        RNN predicts [pred_az_norm, pred_psi_norm].
        Integrates z, vz, az and psi, wz.

    Args:
        state (np.ndarray): Current 18-element state vector.
        u (np.ndarray or list): Control inputs [u1, u2].
        dt (float): Integration time step (s).
        rnn_model (optional): Trained PyTorch model.
        hidden_state (optional): Recurrent hidden state.
        p (SimpleNamespace): Parameters.
        integrate_z (bool): Unused when predicting az directly.

    Returns:
        tuple: (next_state np.ndarray of shape (18,), next_hidden_state)
    """
    z, vz, az = state[0], state[1], state[2]
    psi, wz = state[11], state[16]

    u_arr = np.asarray(u, dtype=np.float32).reshape(-1)

    # =========================================================================
    # GENERAL CASE (Commented out - to be enabled when full 6-DOF RNN is trained)
    # =========================================================================
    # # General case expects 4 inputs: u = [u1, u2, u3, u4]
    # u_norm = (u_arr - p.mean_u) / p.sigma_u
    # u_tensor = torch.tensor([[[float(u_norm[0]), float(u_norm[1]), float(u_norm[2]), float(u_norm[3])]]], dtype=torch.float32)
    # with torch.no_grad():
    #     if hasattr(rnn_model, 'rnn') and hasattr(rnn_model, 'head'):
    #         out, next_hidden = rnn_model.rnn(u_tensor, hidden_state)
    #         pred_tensor = rnn_model.head(out)
    #     else:
    #         pred_tensor, next_hidden = rnn_model(u_tensor, hidden_state)
    #     pred = pred_tensor.detach().cpu().squeeze().numpy()
    #
    # # Full RNN output: [ax, ay, az, wx_dot, wy_dot, wz_dot, phi, psi]
    # ax_phys = float(pred[0]) * p.sigma_ax + p.mean_ax
    # ay_phys = float(pred[1]) * p.sigma_ay + p.mean_ay
    # az_phys = float(pred[2]) * p.sigma_az + p.mean_az
    # wx_dot_phys = float(pred[3]) * p.sigma_wx_dot + p.mean_wx_dot
    # wy_dot_phys = float(pred[4]) * p.sigma_wy_dot + p.mean_wy_dot
    # wz_dot_phys = float(pred[5]) * p.sigma_wz_dot + p.mean_wz_dot
    # phi_phys = float(pred[6]) * p.sigma_phi + p.mean_phi
    # psi_phys = float(pred[7]) * p.sigma_psi + p.mean_psi
    #
    # # Kinematics integration:
    # z_next = z + (vz + az_phys * dt) * dt
    # vz_next = vz + az_phys * dt
    # y_next = state[3] + (state[4] + ay_phys * dt) * dt
    # vy_next = state[4] + ay_phys * dt
    # x_next = state[6] + (state[7] + ax_phys * dt) * dt
    # vx_next = state[7] + ax_phys * dt
    # theta_next = state[9]
    # phi_next = phi_phys
    # psi_next = psi_phys
    # wx_next = state[12] + wx_dot_phys * dt
    # wy_next = state[14] + wy_dot_phys * dt
    # wz_next = state[16] + wz_dot_phys * dt
    #
    # next_state = np.array([
    #     z_next, vz_next, az_phys,
    #     y_next, vy_next, ay_phys,
    #     x_next, vx_next, ax_phys,
    #     theta_next, phi_next, psi_next,
    #     wx_next, wx_dot_phys,
    #     wy_next, wy_dot_phys,
    #     wz_next, wz_dot_phys
    # ], dtype=np.float32)
    # return next_state, next_hidden
    # =========================================================================

    # ── Intermediate Case (Active) ───────────────────────────────────────────
    if rnn_model is not None:
        # Normalize u before feeding into the RNN
        u_norm = (u_arr - p.mean_u) / p.sigma_u
        u_tensor = torch.tensor([[[float(u_norm[0]), float(u_norm[1])]]], dtype=torch.float32)
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
        pred_psi_norm = float(pred[1]) if len(pred) > 1 else 0.0

        # Denormalization to physical units
        az_phys = pred_az_norm * p.sigma_az + p.mean_az
        psi_phys = pred_psi_norm * p.sigma_psi + p.mean_psi
    else:
        next_hidden = hidden_state
        # Baseline analytical model with 2 inputs:
        # Average tension drives vertical acceleration
        u_mean = float(0.5 * (u_arr[0] + u_arr[1]))
        u_phys = u_mean * p.sigma_u + p.mean_u
        az_phys = float(18.3424 * u_phys - p.g)

        # Differential tension drives yaw rotation
        u_diff = float(u_arr[0] - u_arr[1])
        wz_next = float(u_diff * 2.0)
        psi_phys = float(psi + wz_next * dt)

    # Kinematic integration along vertical axis
    vz_next = vz + az_phys * dt
    z_next = z + vz_next * dt
    az_next = az_phys

    # Kinematic integration along yaw axis
    psi_next = psi_phys
    wz_next = (psi_next - psi) / dt
    wz_dot_next = (wz_next - wz) / dt

    next_state = np.zeros(18, dtype=np.float32)
    next_state[0] = z_next
    next_state[1] = vz_next
    next_state[2] = az_next
    # y (3), vy (4), ay (5) remain 0.0
    # x (6), vx (7), ax (8) remain 0.0
    # theta (9), phi (10) remain 0.0
    next_state[11] = psi_next
    # wx (12), wx_dot (13), wy (14), wy_dot (15) remain 0.0
    next_state[16] = wz_next
    next_state[17] = wz_dot_next

    return next_state, next_hidden


def compute_reward(old_state, state, u, outcome, target_ref=(0.0, 0.0), u_prev=None):
    """Computes the step reward based on altitude tracking, yaw orientation, and landing."""
    hubber_loss = lambda x, delta=1.0: 0.5 * x**2 if abs(x) <= delta else delta * (abs(x) - 0.5 * delta)

    if outcome == "landed":
        return 1000.0
    if outcome in ("crashed", "lost", "timeout"):
        return -1000.0

    z_ref, psi_ref = target_ref
    z = state[0]
    vz = state[1]
    theta = state[9]
    phi = state[10]
    psi = state[11]
    wz = state[16]

    # Altitude tracking penalty
    r_z = -10.0 * hubber_loss(z - z_ref)

    # Yaw angle tracking penalty (normalized to [-pi, pi])
    psi_err = (psi - psi_ref + np.pi) % (2.0 * np.pi) - np.pi
    r_psi = -10.0 * hubber_loss(psi_err)

    # Off-axis angular deviation penalties
    r_angles = -2.0 * (abs(theta) + abs(phi))

    # Velocity penalties
    r_vel = -1.0 * hubber_loss(vz) - 0.5 * hubber_loss(wz)

    # Control effort penalty
    u_arr = np.asarray(u, dtype=np.float32)
    r_act = -1.0 * float(np.sum(u_arr ** 2))

    # Smoothness penalty
    if u_prev is not None:
        u_prev_arr = np.asarray(u_prev, dtype=np.float32)
        r_smooth = -1.0 * float(np.sum(np.abs(u_arr - u_prev_arr)))
    else:
        r_smooth = 0.0

    return float(r_z + r_psi + r_angles + r_vel + r_act + r_smooth)


class DroneEnv(gym.Env):
    """Gymnasium environment for 3D drone flight, yaw rotation, and landing."""
    metadata = {"render_modes": []}

    def __init__(self, rnn_model=None, params=PARAMS, max_steps=MAX_STEPS, integrate_z=False, mission=None):
        super().__init__()
        self.p = params
        self.max_steps = max_steps
        self.rnn_model = rnn_model
        self.integrate_z = integrate_z
        self.mission = mission if mission is not None else default_mission
        if not callable(self.mission):
            raise TypeError("mission must be callable: mission(t) -> (z_ref, psi_ref)")

        # Intermediate case (active): 2 control tensions [u1, u2] in [-3.0, 3.0]
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(2,), dtype=np.float32)

        # =====================================================================
        # GENERAL CASE (Commented out - 4 control tensions for 4 rotors)
        # =====================================================================
        # self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(4,), dtype=np.float32)
        # =====================================================================

        # Observation: 18 physical/tracking states:
        # [z - z_ref, vz, az,
        #  y - y_ref, vy, ay,
        #  x - x_ref, vx, ax,
        #  theta, phi, psi - psi_ref,
        #  wx, wx_dot,
        #  wy, wy_dot,
        #  wz, wz_dot]
        high = np.array([
            Z_MAX, 20.0, 100.0,        # z, vz, az
            10.0, 20.0, 100.0,         # y, vy, ay
            10.0, 20.0, 100.0,         # x, vx, ax
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
        self.u_prev = np.zeros(2, dtype=np.float32)
        self.az_prev = -self.p.g

    def _obs(self):
        """Returns the 18-element observation vector with tracking errors."""
        obs = self.state.copy().astype(np.float32)
        z_ref, psi_ref = self.mission(self.t)
        obs[0] -= z_ref
        obs[11] -= psi_ref
        return obs

    def outcome(self):
        """Determines the flight status: 'flying', 'landed', 'crashed', 'lost', or 'timeout'."""
        z = self.state[0]
        vz = self.state[1]
        z_ref, _ = self.mission(self.t)
        if z <= Z_LAND:
            soft = abs(vz) <= SAFE_VZ
            landing_phase = (z_ref == Z_LAND)
            return "landed" if soft and landing_phase else "crashed"
        if z > Z_MAX:
            return "lost"
        if self.steps >= self.max_steps:
            return "timeout"
        return "flying"

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

        z_ref, psi_ref = self.mission(self.t)
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
        """Performs one environment step under control input u."""
        u = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -3.0, 3.0)
        if len(u) == 1:
            u = np.array([u[0], u[0]], dtype=np.float32)
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

        target_ref = self.mission(self.t)
        reward = compute_reward(old, self.state, u, status, target_ref=target_ref, u_prev=self.u_prev)

        terminated = (status != "flying")
        truncated = False  # The mission deadline is a terminal failure, not a rollout cutoff.

        info = {
            "outcome": status,
            "m": {
                "t": self.t,
                "z": float(self.state[0]),
                "vz": float(self.state[1]),
                "az_prev": float(self.az_prev),
                "z_ref": float(target_ref[0]),
                "u_prev": float(self.u_prev[0]),
                "psi": float(self.state[11]),
                "psi_ref": float(target_ref[1]),
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
    """Evaluates mean return and landing success rate over n episodes."""
    returns, landed = [], 0
    for i in range(n):
        _, _, total, outcome = run_episode(env, policy, seed=seed + i)
        returns.append(total)
        landed += (outcome == "landed")
    returns = np.array(returns)
    print(f"mean return: {returns.mean():7.1f} ± {returns.std():5.1f} | landed: {landed}/{n}")
    return returns.mean(), landed / n
