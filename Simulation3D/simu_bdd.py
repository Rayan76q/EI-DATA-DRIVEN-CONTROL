import csv
import numpy as np
from Simulation3D.calculate_coordinates import calculate_coordinates

def simu_bdd(i, bdd_name, type_of_bdd):#type_of_bdd = 1 with only a_z and =2 with a_x, a_y, a_z, w_dot_x, w_dot_y, w_dot_z ; type_of_bdd = 3 ->v_x, v_y, v_z, w_x, w_y, w_z
    return (calculate_coordinates(get_derivatives(i, bdd_name, type_of_bdd), type_of_bdd))

def get_derivatives(i, bdd_name, type_of_bdd):        
    with open(f'{bdd_name}.csv') as fd:
        reader=csv.reader(fd)
        for idx, row in enumerate(reader):
            if idx == i:
                row_i = row
                break
            
    if type_of_bdd == 2:
        len_deriv = 6
    elif type_of_bdd == 1:
        len_deriv = 1
    elif type_of_bdd == 3:
        len_deriv = 6
        row_i = [row_i[idx] for idx in range(len(row_i)) if (idx%18 in (3,4,5,12,13,14))]
    else:
        return

    derivatives = np.zeros((len(row_i)//len_deriv, len_deriv))
    for idx, val in enumerate(row_i):
        derivatives[idx//len_deriv, idx%len_deriv] = float(val)
        
    return derivatives