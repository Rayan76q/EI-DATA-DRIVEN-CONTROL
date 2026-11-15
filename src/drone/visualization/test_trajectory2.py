"""Acceleration profiles that exercise the trajectory integrator."""

import numpy as np

from .calculate_coordinates import calculate_coordinates
from .config import TE


def test_trajectory2():
    """Integrate a climb followed by roll, pitch and yaw pulses."""
    sequence_derivatives = generate_derivatives_all_angles()
    return calculate_coordinates(sequence_derivatives, type_of_bdd=2)


def generate_derivatives(duration=10.0, mode="profile", seed=0):
    """Return an (N, 6) array of [wdotx, wdoty, wdotz, ax, ay, az] samples.

    ``mode="profile"`` gives a climb and a pitch pulse; ``mode="random"`` gives
    seeded Gaussian noise.
    """
    N = int(round(duration / TE))
    t = np.arange(N) * TE
    derivatives = np.zeros((N, 6))

    if mode == "profile":
        # Climb: az = +1 for 2 s, then -1 for 2 s (vz returns to 0).
        derivatives[(t >= 1) & (t < 3), 5] = 1.0
        derivatives[(t >= 3) & (t < 5), 5] = -1.0
        # Pitch pulse on wdoty (wy returns to 0).
        derivatives[(t >= 6) & (t < 6.5), 1] = 0.5
        derivatives[(t >= 6.5) & (t < 7), 1] = -0.5
    elif mode == "random":
        rng = np.random.default_rng(seed)
        std = np.array([0.2, 0.2, 0.2, 0.5, 0.5, 1.0])  # One standard deviation per column.
        derivatives = rng.normal(0.0, std, size=(N, 6))
    else:
        raise ValueError("mode must be 'profile' or 'random'")

    return derivatives


def generate_derivatives_all_angles(duration=10.0):
    """Return an (N, 6) array: a climb, then one pulse pair around each body axis."""
    N = int(round(duration / TE))
    t = np.arange(N) * TE
    d = np.zeros((N, 6))

    # Climb: az = +1 for 1 s, then -1 for 1 s.
    d[(t >= 1) & (t < 2), 5] = 1.0
    d[(t >= 2) & (t < 3), 5] = -1.0
    # Roll pulse (wdotx): +0.4 for 0.5 s, then -0.4 for 0.5 s.
    d[(t >= 3) & (t < 3.5), 0] = 0.4
    d[(t >= 3.5) & (t < 4), 0] = -0.4
    # Pitch pulse (wdoty).
    d[(t >= 4) & (t < 4.5), 1] = 0.4
    d[(t >= 4.5) & (t < 5), 1] = -0.4
    # Yaw pulse (wdotz): 1 s each way.
    d[(t >= 5) & (t < 6), 2] = 0.4
    d[(t >= 6) & (t < 7), 2] = -0.4

    return d
