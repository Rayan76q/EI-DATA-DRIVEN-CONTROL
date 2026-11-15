"""Acceleration profiles for checking the example pose integrator."""

import numpy as np

from .config import TE
from .from_derivatives_to_coordinates import from_derivatives_to_coordinates


def test_trajectory2():
    """Integrate a climb followed by roll, pitch and yaw pulses."""
    derivatives = generate_derivatives_all_angles()
    return from_derivatives_to_coordinates(derivatives, type_of_bdd=2)


def generate_derivatives(duration=10.0, mode="profile", seed=0):
    """Return [wdotx,wdoty,wdotz,ax,ay,az] samples for a profile or seeded noise."""
    count = int(round(duration / TE))
    time = np.arange(count) * TE
    derivatives = np.zeros((count, 6))
    if mode == "profile":
        # Paired pulses return the corresponding velocity to zero.
        derivatives[(time >= 1) & (time < 3), 5] = 1.0
        derivatives[(time >= 3) & (time < 5), 5] = -1.0
        derivatives[(time >= 6) & (time < 6.5), 1] = 0.5
        derivatives[(time >= 6.5) & (time < 7), 1] = -0.5
    elif mode == "random":
        rng = np.random.default_rng(seed)
        scale = np.array([0.2, 0.2, 0.2, 0.5, 0.5, 1.0])
        derivatives = rng.normal(0.0, scale, size=(count, 6))
    else:
        raise ValueError("Mode must be 'profile' or 'random'")
    return derivatives


def generate_derivatives_all_angles(duration=10.0):
    """Return a climb and one paired acceleration pulse around each body axis."""
    count = int(round(duration / TE))
    time = np.arange(count) * TE
    derivatives = np.zeros((count, 6))
    derivatives[(time >= 1) & (time < 2), 5] = 1.0
    derivatives[(time >= 2) & (time < 3), 5] = -1.0
    derivatives[(time >= 3) & (time < 3.5), 0] = 0.4
    derivatives[(time >= 3.5) & (time < 4), 0] = -0.4
    derivatives[(time >= 4) & (time < 4.5), 1] = 0.4
    derivatives[(time >= 4.5) & (time < 5), 1] = -0.4
    derivatives[(time >= 5) & (time < 6), 2] = 0.4
    derivatives[(time >= 6) & (time < 7), 2] = -0.4
    return derivatives
