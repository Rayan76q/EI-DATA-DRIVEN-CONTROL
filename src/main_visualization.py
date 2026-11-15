"""Replay random dataset trajectories in the Meshcat 3D viewer.

Run from the repository root with ``python src/main_visualization.py``. Each
pass picks one of the first 1000 rows of ``db_regime6_helico_output.csv``, read
from the current directory, integrates its velocities and animates the flight.
Stop with Ctrl+C.
"""

import time

import numpy as np

from drone.visualization import (
    add_moving_grid,
    animate_episode,
    creation_modele_drone,
    initialize_visualizer,
    set_axes,
    simu_bdd,
    test_trajectory,
    test_trajectory2,
)


def main():
    """Open the viewer with the textured drone and replay flights in a loop."""
    vis = initialize_visualizer()
    set_axes(vis, 5, 0.03)
    add_moving_grid(vis)
    creation_modele_drone(vis, modele_drone=3)

    # Other sources, for a single flight instead of the loop below.
    # The last argument of simu_bdd is the dataset type: 1 -> az;
    # 2 -> [wdotx, wdoty, wdotz, ax, ay, az]; 3 -> [vx, vy, vz, wx, wy, wz].
    # data = test_trajectory()
    # data = test_trajectory2()
    # data = simu_bdd(4, "bdd_out_mat_05", 1)
    # data = simu_bdd(0, "trajectory_az", 1)
    # data = simu_bdd(10, "db_regime_z_output", 3)
    while True:
        line_idx = np.random.randint(0, 1000)
        print(line_idx)
        # data = simu_bdd(line_idx, "db_4regimes_output", 3)
        data = simu_bdd(line_idx, "db_regime6_helico_output", 3)
        animate_episode(data, vis)
        time.sleep(0.05 * 160)  # Wait 8 s (160 samples) before the next flight.

    # Reached only when the loop is commented out: keeps the viewer open.
    input("Press Enter to stop the simulation.")


if __name__ == "__main__":
    main()
