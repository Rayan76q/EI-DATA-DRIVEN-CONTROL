from pathlib import Path
from drone.visualization import initialize_visualizer, set_axes, creation_modele_drone, animate_episode, test_trajectory, test_trajectory2, simu_bdd, add_moving_grid

if __name__ == "__main__":
    vis = initialize_visualizer()
    set_axes(vis, 5, 0.03)
    add_moving_grid(vis)
    creation_modele_drone(vis)
    #data = test_trajectory()
    #data = test_trajectory2()
    data = simu_bdd(4, str(Path(__file__).resolve().parents[3] / "data/initial_data/bdd_out_mat_05"), 1)#last argument = type of bdd -> 1 if returns a_z else 2
    animate_episode(data, vis) 
    
    input("Stopper la simulation (appuyer sur entrer)")