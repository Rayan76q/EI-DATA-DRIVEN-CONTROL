"""Drone flying with rotation, as a Gymnasium environment (TD 4 / Advanced Tasks).

The drone is a rigid body in 3D space with 18 state variables:
    z, vz, az          (vertical axis translation)
    y, vy, ay          (lateral axis translation)
    x, vx, ax          (longitudinal axis translation)
    theta, phi, psi    (Euler angles: pitch, roll, yaw)
    wx, wx_dot         (angular velocity and acceleration around roll axis)
    wy, wy_dot         (angular velocity and acceleration around pitch axis)
    wz, wz_dot         (angular velocity and acceleration around yaw axis)

Quadrotor control (Symmetric Differential Formulation):
    - Action space: 2 control inputs [u_base, u_diff] in [-3.0, 3.0]^2:
        * u[0] (u_base): collective thrust applied to all 4 motors.
        * u[1] (u_diff): differential yaw command using the configured motor signs.
    - 4 motor commands sent to RNN:
        u_motors = clip(u_base + u_diff * yaw_motor_signs, -3, 3)
    - RNN predicts: [az, w_dot_x, w_dot_y, w_dot_z].

Observation space (11 variables; theta tracking disabled):
    [z - z_ref, vz, az,
     phi - phi_ref, psi - psi_ref,
     wx, wx_dot,
     wy, wy_dot,
     wz, wz_dot]
"""

from types import SimpleNamespace
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

# Training normalization for rnn_model_z_and_angles, supplied per channel.
# Input order: [u1, u2, u3, u4]. Normalize after allocating motor commands.
MEAN_U = np.array([0.0230, 0.0227, 0.0228, 0.0230], dtype=np.float32)
SIGMA_U = np.array([1.7386, 1.7386, 1.7389, 1.7389], dtype=np.float32)
# Output order: [az, wx_dot, wy_dot, wz_dot].
MEAN_AZ = -9.7264
SIGMA_AZ = 17.3450
MEAN_WX_DOT = -2.5107e-03
SIGMA_WX_DOT = 1.0602
MEAN_WY_DOT = -1.5441e-02
SIGMA_WY_DOT = 1.0702
MEAN_WZ_DOT = 4.0882e-04
SIGMA_WZ_DOT = 0.6968

# Motor order is [u1, u2, u3, u4]. Probing the learned yaw acceleration gives
# positive response for motors 1 & 3 and negative for motors 2 & 4. Use these
# model-defined yaw pairs; the visualizer's numbering is not a dynamics map.
YAW_MOTOR_SIGNS = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)

# Physical parameters of the system
PARAMS = SimpleNamespace(
    dt=0.05,            # Sampling period (s)
    g=9.816,            # Gravitational acceleration (m/s²)
    mean_u=MEAN_U,
    sigma_u=SIGMA_U,
    mean_az=MEAN_AZ,
    sigma_az=SIGMA_AZ,
    mean_wx_dot=MEAN_WX_DOT,
    sigma_wx_dot=SIGMA_WX_DOT,
    mean_wy_dot=MEAN_WY_DOT,
    sigma_wy_dot=SIGMA_WY_DOT,
    mean_wz_dot=MEAN_WZ_DOT,
    sigma_wz_dot=SIGMA_WZ_DOT,
    yaw_motor_signs=YAW_MOTOR_SIGNS,
)

# Landing and operational thresholds
Z_LAND = -0.3          # Landing platform altitude along vertical axis (m)
Z_MAX = 10.0           # Ceiling altitude limit (m)
SAFE_VZ = 0.5          # Maximum vertical touchdown speed for soft landing (m/s)
SAFE_VX = 0.5
SAFE_VY = 0.5
SAFE_TILT = np.deg2rad(10.0)  # Maximum roll error at touchdown (rad)
SAFE_ANGULAR_SPEED = 0.5     # Maximum roll/yaw angular-speed norm at touchdown (rad/s)
MAX_STEPS = 200        # Mission length: 10 s at Ts = 0.05 s


def default_mission(t):
    """Return (z_ref, phi_ref, psi_ref); theta tracking is disabled."""
    if t < 2.0:
        # return (0.0, 0.0, 0.0, 0.0)  # With theta_ref
        return (0.0, 0.0, 0.0)       # Step 1: On the platform (z = 0)
    elif t < 5.0:
        # return (1.0, 0.0, 0.0, 0.0)  # With theta_ref
        return (1.0, 0.0, 0.0)       # Step 2: Climb and hold altitude (z = 1.0)
    elif t < 8.0:
        # return (1.0, 0.0, 0.0, float(2.0 * np.pi * (t - 5.0) / 3.0))
        return (1.0, 0.0, float(2.0 * np.pi * (t - 5.0) / 3.0))
    else:
        # return (Z_LAND, 0.0, 0.0, float(2.0 * np.pi))  # With theta_ref
        return (Z_LAND, 0.0, float(2.0 * np.pi))  # Land holding the completed turn


def mission_reference(reference):
    """Read (z_ref, phi_ref, psi_ref); ignore theta in older four-value missions."""
    if len(reference) == 2:
        z_ref, psi_ref = reference
        return z_ref, 0.0, psi_ref
    if len(reference) == 3:
        return tuple(reference)
    if len(reference) == 4:
        z_ref, _theta_ref, phi_ref, psi_ref = reference
        return z_ref, phi_ref, psi_ref
    raise ValueError("mission must return (z_ref, phi_ref, psi_ref)")


# Backward-compatible import for notebooks using the original function name.
mission = default_mission


def dynamics(state, u, dt=0.05, rnn_model=None, hidden_state=None, p=PARAMS, integrate_z=False):
    """Calculates next 18-variable state and next recurrent hidden state using the RNN.

    Args:
        state (np.ndarray): Current 18-element state vector.
        u (np.ndarray or list): 2-element control input [u_base, u_diff].
        dt (float): Integration time step (s).
        rnn_model: Trained PyTorch model predicting (az, w_dot_x, w_dot_y, w_dot_z).
        hidden_state: Recurrent hidden state tensor.
        p (SimpleNamespace): Parameters.
        integrate_z (bool): Unused when predicting az directly.

    Returns:
        tuple: (next_state np.ndarray of shape (18,), next_hidden_state)
    """
    if rnn_model is None:
        raise ValueError("dynamics requires an rnn_model. Fallback has been disabled.")

    z, vz, az = state[0], state[1], state[2]
    theta, phi, psi = state[9], state[10], state[11]
    wx, wx_dot = state[12], state[13]
    wy, wy_dot = state[14], state[15]
    wz, wz_dot = state[16], state[17]

    u_arr = np.asarray(u, dtype=np.float32).reshape(-1)
    if len(u_arr) < 2:
        # A scalar command is collective-only: the missing differential is zero.
        u_arr = np.pad(u_arr, (0, 2 - len(u_arr)), mode='constant')

    # Allocate in the RNN's motor order and enforce each motor's command limit.
    u_base = float(u_arr[0])
    u_diff = float(u_arr[1])
    u_motors = np.clip(u_base + u_diff * p.yaw_motor_signs, -3.0, 3.0)

    # Normalize 4 motor inputs before entering the RNN
    u_norm = (u_motors - p.mean_u) / p.sigma_u
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
    # Expected RNN outputs: [az, w_dot_x, w_dot_y, w_dot_z]
    pred_az_norm = float(pred[0])
    pred_wx_dot_norm = float(pred[1]) if len(pred) > 1 else 0.0
    pred_wy_dot_norm = float(pred[2]) if len(pred) > 2 else 0.0
    pred_wz_dot_norm = float(pred[3]) if len(pred) > 3 else 0.0

    sigma_wx_dot = getattr(p, 'sigma_wx_dot', 1.0) or 1.0
    mean_wx_dot = getattr(p, 'mean_wx_dot', 0.0) or 0.0
    sigma_wy_dot = getattr(p, 'sigma_wy_dot', 1.0) or 1.0
    mean_wy_dot = getattr(p, 'mean_wy_dot', 0.0) or 0.0
    sigma_wz_dot = getattr(p, 'sigma_wz_dot', 1.0) or 1.0
    mean_wz_dot = getattr(p, 'mean_wz_dot', 0.0) or 0.0

    # Denormalization to physical units
    az_phys = pred_az_norm * p.sigma_az + p.mean_az
    wx_dot_phys = pred_wx_dot_norm * sigma_wx_dot + mean_wx_dot
    wy_dot_phys = pred_wy_dot_norm * sigma_wy_dot + mean_wy_dot
    wz_dot_phys = pred_wz_dot_norm * sigma_wz_dot + mean_wz_dot

    # 1. Linear kinematics along vertical axis
    vz_next = vz + az_phys * dt
    z_next = z + vz_next * dt
    az_next = az_phys

    # 2. Angular accelerations
    wx_dot_next = wx_dot_phys
    wy_dot_next = wy_dot_phys
    wz_dot_next = wz_dot_phys

    # 3. Angular rates integration
    wx_next = wx + wx_dot_phys * dt
    wy_next = wy + wy_dot_phys * dt
    wz_next = wz + wz_dot_phys * dt

    # 4. Attitude propagation (Euler angle kinematic rates)
    phi_dot = wx + np.sin(phi) * np.tan(theta) * wy + np.cos(phi) * np.tan(theta) * wz
    theta_dot = np.cos(phi) * wy - np.sin(phi) * wz
    cos_theta = np.cos(theta)

    if abs(cos_theta) > 1e-8:
        psi_dot = (np.sin(phi) * wy + np.cos(phi) * wz) / cos_theta
    else:
        psi_dot = wz_next

    phi_next = float(phi + phi_dot * dt)
    theta_next = float(theta + theta_dot * dt)
    psi_next = float(psi + psi_dot * dt)

    next_state = np.zeros(18, dtype=np.float32)
    next_state[0] = z_next
    next_state[1] = vz_next
    next_state[2] = az_next
    # y (3), vy (4), ay (5) remain 0.0
    # x (6), vx (7), ax (8) remain 0.0
    next_state[9] = theta_next
    next_state[10] = phi_next
    next_state[11] = psi_next
    next_state[12] = wx_next
    next_state[13] = wx_dot_next
    next_state[14] = wy_next
    next_state[15] = wy_dot_next
    next_state[16] = wz_next
    next_state[17] = wz_dot_next

    return next_state, next_hidden


def compute_reward(old_state, state, u, outcome, target_ref=(0.0, 0.0), u_prev=None, dt=PARAMS.dt):
    """Integrate running costs over dt; keep terminal success/failure rewards fixed."""
    hubber_loss = lambda x, delta=1.0: 0.5 * x**2 if abs(x) <= delta else delta * (abs(x) - 0.5 * delta)

    if isinstance(outcome, tuple):
        z_status, psi_status = outcome
    else:
        z_status, psi_status = outcome, "has_turned"

    if z_status == "landed":
        return 1000.0 if psi_status == "has_turned" else -1000.0
    if z_status in ("crashed", "lost", "timeout"):
        return -1000.0

    z_ref, phi_ref, psi_ref = mission_reference(target_ref)
    z = state[0]
    vz = state[1]
    # theta = state[9]  # Pitch tracking disabled.
    phi = state[10]
    psi = state[11]
    wz = state[16]

    # Altitude tracking penalty
    r_z = -20.0 * hubber_loss(z - z_ref)

    # Track accumulated positive rotation, keeping 0 and 2π distinct.
    psi_err = psi - psi_ref
    r_psi = -10.0 * hubber_loss(psi_err)

    # Off-axis angular deviation penalties
    # r_angles = -10.0 * (abs(theta - theta_ref) + abs(phi - phi_ref))
    r_angles = -10.0 * abs(phi - phi_ref)

    # Velocity penalties
    r_vel = -1.0 * hubber_loss(vz) - 0.5 * hubber_loss(wz)

    # Penalize differential effort so collective thrust remains the cheaper lift input.
    u_arr = np.asarray(u, dtype=np.float32).reshape(-1)

    # The second element of u_arr is the differential effort, which should be penalized.
    r_act = -1.0 * float(u_arr[1] ** 2)

    # Smoothness penalty
    if u_prev is not None:
        u_prev_arr = np.asarray(u_prev, dtype=np.float32)
        r_smooth = -1.0 * float(np.sum(np.abs(u_arr - u_prev_arr)))
    else:
        r_smooth = 0.0


    return float(dt * (r_z + r_psi + r_angles + r_vel + r_act + r_smooth))


class DroneEnv(gym.Env):
    """Gymnasium environment for 3D drone flight, yaw rotation, and landing."""
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
        self.p = params
        self.max_steps = max_steps
        self.rnn_model = rnn_model
        self.integrate_z = integrate_z
        self.mission = mission if mission is not None else default_mission
        if not callable(self.mission):
            raise TypeError("mission must be callable: mission(t) -> (z_ref, phi_ref, psi_ref)")

        # Intermediate case: 2 control inputs [u_base, u_diff] in [-3.0, 3.0]^2
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(2,), dtype=np.float32)

        # Observation: 11 physical/tracking states:
        # [z - z_ref, vz, az,
        #  phi - phi_ref, psi - psi_ref,
        #  wx, wx_dot,
        #  wy, wy_dot,
        #  wz, wz_dot]
        high = np.array([
            Z_MAX, 20.0, 100.0,        # z, vz, az
            # np.pi,                  # theta tracking disabled
            np.pi, 2.0 * np.pi,        # phi, psi
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
        self.has_turned = False
        self.out = self.__class__.out

    def _obs(self):
        """Returns the 11-element observation vector; theta tracking is disabled."""
        z_ref, phi_ref, psi_ref = mission_reference(self.mission(self.t))
        obs = np.array([
            self.state[0] - z_ref,
            self.state[1],
            self.state[2],
            # self.state[9] - theta_ref,
            self.state[10] - phi_ref,
            self.state[11] - psi_ref,
            self.state[12],
            self.state[13],
            self.state[14],
            self.state[15],
            self.state[16],
            self.state[17],
        ], dtype=np.float32)
        return obs

    def outcome(self):
        """Return (z status, yaw status), remembering a completed positive 2π turn."""
        z = self.state[0]
        psi = self.state[11]
        vz = self.state[1]
        z_ref, phi_ref, _ = mission_reference(self.mission(self.t))

        self.has_turned = self.has_turned or bool(psi >= 2.0 * np.pi)
        psi_status = self.out.has_turned if self.has_turned else self.out.no_turn
        if z <= Z_LAND:
            soft = abs(vz) <= SAFE_VZ
            # level = (abs(self.state[9] - theta_ref) <= SAFE_TILT
            #          and abs(self.state[10] - phi_ref) <= SAFE_TILT)
            level = abs(self.state[10] - phi_ref) <= SAFE_TILT
            # settled = np.linalg.norm(self.state[[12, 14, 16]]) <= SAFE_ANGULAR_SPEED
            settled = np.linalg.norm(self.state[[12, 16]]) <= SAFE_ANGULAR_SPEED
            landing_phase = (z_ref == Z_LAND)
            z_status = self.out.landed if soft and level and settled and landing_phase else self.out.crashed
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
                raise TypeError("mission must be callable: mission(t) -> (z_ref, phi_ref, psi_ref)")
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

        z_ref, phi_ref, psi_ref = mission_reference(self.mission(self.t))
        info = {
            "outcome": self.outcome(),
            "m": {
                "t": self.t,
                "z": float(self.state[0]),
                "vz": float(self.state[1]),
                "az_prev": float(self.az_prev),
                "z_ref": float(z_ref),
                "theta": float(self.state[9]),
                # "theta_ref": float(theta_ref),
                "phi": float(self.state[10]),
                "phi_ref": float(phi_ref),
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
        status = self.outcome()

        target_ref = self.mission(self.t)
        reward = compute_reward(old, self.state, u, status, target_ref=target_ref, u_prev=self.u_prev, dt=self.p.dt)
        z_ref, phi_ref, psi_ref = mission_reference(target_ref)

        terminated = (status[0] != self.out.flying)
        truncated = False  # The mission deadline is a terminal failure, not a rollout cutoff.

        info = {
            "outcome": status,
            "m": {
                "t": self.t,
                "z": float(self.state[0]),
                "vz": float(self.state[1]),
                "az_prev": float(self.az_prev),
                "z_ref": float(z_ref),
                "theta": float(self.state[9]),
                # "theta_ref": float(theta_ref),
                "phi": float(self.state[10]),
                "phi_ref": float(phi_ref),
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
        is_landed = (outcome[0] == "landed" if isinstance(outcome, tuple) else outcome == "landed")
        landed += int(is_landed)
    returns = np.array(returns)
    print(f"mean return: {returns.mean():7.1f} ± {returns.std():5.1f} | landed: {landed}/{n}")
    return returns.mean(), landed / n
