import numpy as np
from Simulation3D.calculate_coordinates import calculate_coordinates
from Simulation3D.config import TE

def test_trajectory2():
    sequence_derivatives = generate_derivatives_all_angles()
    return (calculate_coordinates(sequence_derivatives, type_of_bdd=2))

def generate_derivatives(duration=10.0, mode="profile", seed=0):
    """Return an array of shape (N, 6), one row per time step.
    Columns: [w_dot_x, w_dot_y, w_dot_z, a_x, a_y, a_z]  (same order as calculate_next_state)
    """
    N = int(round(duration / TE))
    t = np.arange(N) * TE
    derivatives = np.zeros((N, 6))

    if mode == "profile":
        # climb: a_z = +1 for 2 s, then -1 for 2 s (v_z returns to 0)
        derivatives[(t >= 1) & (t < 3), 5] = 1.0
        derivatives[(t >= 3) & (t < 5), 5] = -1.0
        # pitch pulse on w_dot_y (w_y returns to 0)
        derivatives[(t >= 6) & (t < 6.5), 1] = 0.5
        derivatives[(t >= 6.5) & (t < 7), 1] = -0.5

    elif mode == "random":
        rng = np.random.default_rng(seed)
        std = np.array([0.2, 0.2, 0.2, 0.5, 0.5, 1.0])   # one std per column
        derivatives = rng.normal(0.0, std, size=(N, 6))

    else:
        raise ValueError("mode must be 'profile' or 'random'")

    return derivatives
    
    
def generate_derivatives_all_angles(duration=10.0):
    """Return an array of shape (N, 6), one row per time step.
    Columns: [w_dot_x, w_dot_y, w_dot_z, a_x, a_y, a_z]
    """
    N = int(round(duration / TE))
    t = np.arange(N) * TE
    d = np.zeros((N, 6))

    # climb: a_z = +1 for 2 s, then -1 for 2 s
    d[(t >= 1) & (t < 2), 5] = 1.0
    d[(t >= 2) & (t < 3), 5] = -1.0

    # roll pulse (w_dot_x): +0.4 for 0.5 s, then -0.4 for 0.5 s
    d[(t >= 3) & (t < 3.5), 0] = 0.4
    d[(t >= 3.5) & (t < 4), 0] = -0.4

    # pitch pulse (w_dot_y)
    d[(t >= 4) & (t < 4.5), 1] = 0.4
    d[(t >= 4.5) & (t < 5), 1] = -0.4

    # yaw pulse (w_dot_z): 1 s each way
    d[(t >= 5) & (t < 6), 2] = 0.4
    d[(t >= 6) & (t < 7), 2] = -0.4

    return d