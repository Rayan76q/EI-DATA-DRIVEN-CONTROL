"""Turn a sequence of derivative samples into the viewer's pose rows."""

import numpy as np

from .calculate_next_state import calculate_next_state
from .config import INITIAL_STATE, TE


def calculate_coordinates(sequence_derivatives, type_of_bdd):
    """Integrate from INITIAL_STATE and return rows [t,x,y,z,phi,theta,psi].

    The first row is the initial pose at t=0, followed by one row per sample.
    """
    def pose(state, time):
        return np.array([time, state[0], state[1], state[2], state[6], state[7], state[8]])

    state = INITIAL_STATE.copy()
    sequence_coordinates = [pose(state, 0)]
    for i, derivatives in enumerate(sequence_derivatives):
        state = calculate_next_state(state, derivatives, type_of_bdd)
        sequence_coordinates.append(pose(state, (i + 1) * TE))
    return np.array(sequence_coordinates)
