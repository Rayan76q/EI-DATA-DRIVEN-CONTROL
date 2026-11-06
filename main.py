import numpy as np
import time
from Simulation3D import initialize_visualizer, set_axes, creation_modele_drone, animate_episode, test_trajectory, test_trajectory2, simu_bdd, add_moving_grid

if __name__ == "__main__":
    vis = initialize_visualizer()
    set_axes(vis, 5, 0.03)
    add_moving_grid(vis)
    creation_modele_drone(vis, modele_drone=3)
    #data = test_trajectory()
    #data = test_trajectory2()
    #data = simu_bdd(4, "bdd_out_mat_05", 1)#last argument = type of bdd -> 1 if returns a_z; 2 -> a_x, a_y, a_z, w_dot_x, w_dot_y, w_dot_z ; 3 -> v_x, v_y, v_z, w_x, w_y, w_z
    #data = simu_bdd(0, "trajectory_az", 1)
    #data = simu_bdd(10, "db_regime_z_output", 3)
    while True:
        line_idx = np.random.randint(0, 1000)
        print(line_idx)
        #data = simu_bdd(line_idx, "db_4regimes_output", 3)
        data = simu_bdd(line_idx, "db_regime6_helico_output", 3)
        animate_episode(data, vis) 
        time.sleep(0.05 * 160)
    
    input("Stopper la simulation (appuyer sur entrer)")