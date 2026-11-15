"""Read one trajectory of a CSV dataset and integrate it into viewer poses."""

import csv

import numpy as np

from .calculate_coordinates import calculate_coordinates

PLANT_STATE_COUNT = 18
# Positions of vx, vy, vz, wx, wy, wz among the 18 interleaved plant states.
VELOCITY_COLUMNS = (3, 4, 5, 12, 13, 14)


def simu_bdd(i, bdd_name, type_of_bdd):
    """Return rows [t,x,y,z,phi,theta,psi] for row ``i`` of ``<bdd_name>.csv``."""
    return calculate_coordinates(get_derivatives(i, bdd_name, type_of_bdd), type_of_bdd)


def get_derivatives(i, bdd_name, type_of_bdd):
    """Load row ``i`` of ``<bdd_name>.csv`` as one array row per time step.

    The path is relative to the current directory. Dataset types:

    1. one value per step: az;
    2. six values per step: [wdotx, wdoty, wdotz, ax, ay, az];
    3. the 18 interleaved plant states, as in the ``<mode>_outputs.csv`` files
       of ``generate_training_data``; [vx, vy, vz, wx, wy, wz] are kept.

    Other types return None.
    """
    with open(f"{bdd_name}.csv") as fd:
        reader = csv.reader(fd)
        for idx, row in enumerate(reader):
            if idx == i:
                row_i = row
                break

    if type_of_bdd == 1:
        len_deriv = 1
    elif type_of_bdd == 2:
        len_deriv = 6
    elif type_of_bdd == 3:
        len_deriv = 6
        row_i = [value for idx, value in enumerate(row_i) if idx % PLANT_STATE_COUNT in VELOCITY_COLUMNS]
    else:
        return None

    derivatives = np.zeros((len(row_i) // len_deriv, len_deriv))
    for idx, value in enumerate(row_i):
        derivatives[idx // len_deriv, idx % len_deriv] = float(value)
    return derivatives
