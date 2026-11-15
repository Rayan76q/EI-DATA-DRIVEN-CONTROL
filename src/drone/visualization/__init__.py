"""Meshcat 3D viewer for drone trajectories.

Run ``python src/main_visualization.py`` to replay dataset trajectories. Poses
are rows [t, x, y, z, phi, theta, psi], built by ``test_trajectory``,
``test_trajectory2`` or ``simu_bdd`` from a CSV dataset.
"""

from .animate_episode import animate_episode
from .calculate_coordinates import calculate_coordinates
from .calculate_next_state import calculate_next_state
from .initialize_visualizer import add_moving_grid, initialize_visualizer, set_axes
from .model_3D import creation_modele_drone
from .simu_bdd import simu_bdd
from .test_trajectory import test_trajectory
from .test_trajectory2 import test_trajectory2
