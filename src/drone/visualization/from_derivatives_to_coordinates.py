"""Convert acceleration trajectories into the viewer's pose format."""

import numpy as np

from .calculate_next_state import calculate_next_state
from .config import INITIAL_STATE, TE


def from_derivatives_to_coordinates(sequence_derivatives, type_of_bdd):
    """Return rows [t,x,y,z,phi,theta,psi], including the initial pose."""
    state = INITIAL_STATE.copy()
    coordinates = [[0, *state[:3], *state[6:9]]]
    for index, derivatives in enumerate(sequence_derivatives):
        state = calculate_next_state(state, derivatives, type_of_bdd)
        coordinates.append([(index + 1) * TE, *state[:3], *state[6:9]])
    return np.asarray(coordinates)
