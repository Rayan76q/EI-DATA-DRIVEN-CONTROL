import csv
import numpy as np
from Simulation3D.from_derivatives_to_coordinates import from_derivatives_to_coordinates

def simu_bdd(i, bdd_name, type_of_bdd):#type_of_bdd = 1 with only a_z and =2 with a_x, a_y, a_z, w_dot_x, w_dot_y, w_dot_z
    return (from_derivatives_to_coordinates(get_derivatives(i, bdd_name, type_of_bdd), type_of_bdd))

def get_derivatives(i, bdd_name, type_of_bdd):        
    with open(f'{bdd_name}.csv') as fd:
        reader=csv.reader(fd)
        for idx, row in enumerate(reader):
            if idx == i:
                row_i = row
                break

    if type_of_bdd == 2:
        derivatives = np.array((i//6, i%6))
        for idx, val in enumerate(row_i):
            derivatives[i//6, i%6] = float(val)
    elif type_of_bdd == 1:
        derivatives = np.zeros(len(row_i))
        for idx, val in enumerate(row_i):
            derivatives[idx] = float(val)
    else:
        return
    return derivatives