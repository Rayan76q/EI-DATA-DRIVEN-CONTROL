"""A smooth example flight that does not need a dataset."""

import numpy as np

from .config import TE


def test_trajectory():
    """Return rows [t,x,y,z,phi,theta,psi] for a climb, square, turn and descent."""
    waypoints = np.array([
        # x, y, z, phi, theta, psi
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.5, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.5, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.5, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.5, 0.0, 0.0, 2 * np.pi],
        [0.0, 0.0, 0.5, 0.0, 0.0, 2 * np.pi],
    ])
    segment_duration = 5.0
    steps_per_segment = int(segment_duration / TE)

    trajectory = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        for j in range(steps_per_segment):
            alpha = j / steps_per_segment
            # Smoothstep gives zero velocity at each waypoint.
            alpha_smooth = 3 * alpha**2 - 2 * alpha**3
            trajectory.append(start + alpha_smooth * (end - start))
    trajectory.append(waypoints[-1])

    trajectory = np.array(trajectory)
    t = np.arange(len(trajectory)) * TE
    return np.column_stack((t, trajectory))
