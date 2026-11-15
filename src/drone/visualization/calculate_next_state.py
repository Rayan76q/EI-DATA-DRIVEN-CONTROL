"""One integration step of the 12-component drone state."""

import numpy as np

from .config import TE


def calculate_next_state(state_t, derivatives_t2, type_of_bdd):
    """Advance [x,y,z,vx,vy,vz,phi,theta,psi,wx,wy,wz] by one period TE.

    ``type_of_bdd`` gives the content of ``derivatives_t2``:

    1. [az]: only z and vz move; every other component is set to 0.
    2. [wdotx, wdoty, wdotz, ax, ay, az]: the accelerations update the
       velocities, then the new velocities update the position.
    3. [vx, vy, vz, wx, wy, wz]: these velocities replace the current ones
       and update the position.

    For types 2 and 3, the Euler angles move with the current body rates and
    the current angles are read in degrees. Other types return None.
    """
    x_t, y_t, z_t, v_x_t, v_y_t, v_z_t, phi_t, theta_t, psi_t, w_x_t, w_y_t, w_z_t = state_t

    if type_of_bdd == 1:
        a_z_t = derivatives_t2[0]
        v_z_t2 = v_z_t + a_z_t * TE
        z_t2 = z_t + v_z_t2 * TE
        return (0, 0, z_t2, 0, 0, v_z_t2, 0, 0, 0, 0, 0, 0)

    if type_of_bdd == 2:
        w_dot_x_t, w_dot_y_t, w_dot_z_t, a_x_t, a_y_t, a_z_t = derivatives_t2
        v_x_t2, v_y_t2, v_z_t2 = v_x_t + a_x_t * TE, v_y_t + a_y_t * TE, v_z_t + a_z_t * TE
        w_x_t2, w_y_t2, w_z_t2 = w_x_t + w_dot_x_t * TE, w_y_t + w_dot_y_t * TE, w_z_t + w_dot_z_t * TE
    elif type_of_bdd == 3:
        v_x_t2, v_y_t2, v_z_t2, w_x_t2, w_y_t2, w_z_t2 = derivatives_t2
    else:
        return None

    x_t2, y_t2, z_t2 = x_t + v_x_t2 * TE, y_t + v_y_t2 * TE, z_t + v_z_t2 * TE
    phi_t2, theta_t2, psi_t2 = next_angles(phi_t, theta_t, psi_t, w_x_t, w_y_t, w_z_t)
    return (x_t2, y_t2, z_t2, v_x_t2, v_y_t2, v_z_t2, phi_t2, theta_t2, psi_t2, w_x_t2, w_y_t2, w_z_t2)


def next_angles(phi_t, theta_t, psi_t, w_x_t, w_y_t, w_z_t):
    """Integrate the Euler angle rates obtained from the body rates."""
    phi_rad = phi_t * np.pi / 180
    theta_rad = theta_t * np.pi / 180
    phi_dot_t = w_x_t + np.sin(phi_rad) * np.tan(theta_rad) * w_y_t + np.cos(phi_rad) * np.tan(theta_rad) * w_z_t
    theta_dot_t = np.cos(phi_rad) * w_y_t - np.sin(phi_rad) * w_z_t
    psi_dot_t = (np.sin(phi_rad) * w_y_t + np.cos(phi_rad) * w_z_t) / np.cos(theta_rad)
    return phi_t + phi_dot_t * TE, theta_t + theta_dot_t * TE, psi_t + psi_dot_t * TE
