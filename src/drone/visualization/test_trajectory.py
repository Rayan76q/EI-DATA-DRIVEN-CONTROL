"""A smooth example flight that does not require a dataset."""

import numpy as np

from .config import TE


def test_trajectory():
    """Return [t,x,y,z,phi,theta,psi] for a climb, square, turn and descent."""
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
    for index in range(len(waypoints) - 1):
        start, end = waypoints[index:index + 2]
        for step in range(steps_per_segment):
            alpha = step / steps_per_segment
            # Smoothstep gives zero velocity at each waypoint.
            alpha_smooth = 3 * alpha**2 - 2 * alpha**3
            trajectory.append(start + alpha_smooth * (end - start))

    trajectory.append(waypoints[-1])
    trajectory = np.asarray(trajectory)
    time = np.arange(len(trajectory)) * TE
    return np.column_stack((time, trajectory))
