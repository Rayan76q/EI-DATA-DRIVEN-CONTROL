"""Integrate acceleration samples for the optional trajectory viewer."""

import numpy as np

from .config import TE


def calculate_next_state(state_t, derivatives_t, type_of_bdd):
    """Advance [x,y,z,vx,vy,vz,phi,theta,psi,wx,wy,wz] by one Euler step.

    Dataset type 1 supplies scalar az. Type 2 supplies angular accelerations
    followed by linear accelerations: [wdotx, wdoty, wdotz, ax, ay, az].
    """
    x, y, z, vx, vy, vz, phi, theta, psi, wx, wy, wz = state_t

    if type_of_bdd == 1:
        next_vz = vz + derivatives_t * TE
        next_z = z + vz * TE
        return (0, 0, next_z, 0, 0, next_vz, 0, 0, 0, 0, 0, 0)

    if type_of_bdd != 2:
        raise ValueError("Dataset type must be 1 (az) or 2 (six acceleration channels)")

    wdotx, wdoty, wdotz, ax, ay, az = derivatives_t
    next_vx, next_vy, next_vz = vx + ax * TE, vy + ay * TE, vz + az * TE
    next_x, next_y, next_z = x + vx * TE, y + vy * TE, z + vz * TE
    next_wx, next_wy, next_wz = wx + wdotx * TE, wy + wdoty * TE, wz + wdotz * TE

    phi_dot = wx + np.sin(phi) * np.tan(theta) * wy + np.cos(phi) * np.tan(theta) * wz
    theta_dot = np.cos(phi) * wy - np.sin(phi) * wz
    psi_dot = (np.sin(phi) * wy + np.cos(phi) * wz) / np.cos(theta)
    next_phi = phi + phi_dot * TE
    next_theta = theta + theta_dot * TE
    next_psi = psi + psi_dot * TE

    return (
        next_x, next_y, next_z,
        next_vx, next_vy, next_vz,
        next_phi, next_theta, next_psi,
        next_wx, next_wy, next_wz,
    )
