"""Sampling period and initial state of the trajectory integrator."""

import numpy as np

TE = 0.05  # Sampling period in seconds, as in the datasets.
# [x, y, z, vx, vy, vz, phi, theta, psi, wx, wy, wz]
INITIAL_STATE = np.array([0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
