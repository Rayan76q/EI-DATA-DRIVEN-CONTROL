"""Read a flattened acceleration trajectory for the optional viewer."""

import csv
from pathlib import Path

import numpy as np

from .from_derivatives_to_coordinates import from_derivatives_to_coordinates


def simu_bdd(i, bdd_name, type_of_bdd):
    """Integrate CSV trajectory i and return time, position and Euler angles."""
    derivatives = get_derivatives(i, bdd_name, type_of_bdd)
    return from_derivatives_to_coordinates(derivatives, type_of_bdd)


def get_derivatives(i, bdd_name, type_of_bdd):
    """Load one CSV row as scalar az or [wdotx,wdoty,wdotz,ax,ay,az] samples."""
    path = Path(bdd_name).with_suffix(".csv")
    with path.open() as file:
        row = next((row for index, row in enumerate(csv.reader(file)) if index == i), None)
    if row is None:
        raise IndexError(f"Trajectory {i} is absent from {path}")

    values = np.asarray([float(value) for value in row if value.strip()])
    if type_of_bdd == 1:
        return values
    if type_of_bdd == 2:
        return values.reshape(-1, 6)
    raise ValueError("Dataset type must be 1 (az) or 2 (six acceleration channels)")
