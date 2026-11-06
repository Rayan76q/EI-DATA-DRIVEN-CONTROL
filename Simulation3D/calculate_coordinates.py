import numpy as np
from Simulation3D.calculate_next_state import calculate_next_state
from Simulation3D.config import INITIAL_STATE, TE

def calculate_coordinates(sequence_derivatives, type_of_bdd):
    sequence_coordinates = []#t, x, y, z, phi, theta, psi
    
    def append_coordinates(state, time):
        this_coord  = np.array([time, state[0], state[1], state[2], state[6], state[7], state[8]])
        sequence_coordinates.append(this_coord)
    
    actual_state = INITIAL_STATE.copy()
    append_coordinates(actual_state, 0)
    
    for i, derivatives in enumerate(sequence_derivatives):
        actual_state = calculate_next_state(actual_state, derivatives, type_of_bdd)
        append_coordinates(actual_state, (i+1) * TE)
        
    return np.array(sequence_coordinates)