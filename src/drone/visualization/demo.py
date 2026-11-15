"""Display a generated flight in Meshcat without loading a dataset."""

from .animate_episode import animate_episode
from .initialize_visualizer import add_moving_grid, initialize_visualizer, set_axes
from .model_3D import creation_modele_drone
from .test_trajectory import test_trajectory


def main():
    """Open the viewer and animate a climb, square flight, turn and descent."""
    visualizer = initialize_visualizer()
    set_axes(visualizer, length=5, radius=0.03)
    add_moving_grid(visualizer)
    creation_modele_drone(visualizer)
    animate_episode(test_trajectory(), visualizer)
    input("Press Enter to close the simulation. ")


if __name__ == "__main__":
    main()
