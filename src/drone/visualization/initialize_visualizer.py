"""Create the Meshcat scene and its world reference geometry."""

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np


def initialize_visualizer():
    """Start Meshcat, open it in the browser and print the viewer URL."""
    vis = meshcat.Visualizer()
    vis.open()
    print(vis.url())
    return vis


def set_axes(visualizer, length=5.0, radius=0.5):
    """Draw red, green and blue x, y, z axes that move with the world."""
    visualizer["/Axes"].set_property("visible", False)

    rotations = {
        "x": (0xFF0000, tf.rotation_matrix(-np.pi / 2, [0, 0, 1])),
        "y": (0x00FF00, np.eye(4)),
        "z": (0x0000FF, tf.rotation_matrix(np.pi / 2, [1, 0, 0])),
    }
    shift = tf.translation_matrix([0, length / 2, 0])  # Along the cylinder's own axis.
    for name, (color, R) in rotations.items():
        node = visualizer[f"world/world_axes/{name}"]
        node.set_object(g.Cylinder(length, radius), g.MeshLambertMaterial(color=color))
        node.set_transform(R @ shift)


def add_moving_grid(visualizer, spacing=1.0, n=30, color=0x888888):
    """Draw a ground grid in the world frame instead of Meshcat's fixed grid."""
    visualizer["/Grid"].set_property("visible", False)

    length_grid = n * spacing
    pts = []
    for s in np.arange(-n, n + 1) * spacing:
        # One line parallel to y, then one parallel to x.
        pts += [[s, -length_grid, 0], [s, length_grid, 0], [-length_grid, s, 0], [length_grid, s, 0]]
    verts = np.array(pts, dtype=np.float32).T  # Shape (3, 2 * number_of_lines).
    visualizer["world/grid"].set_object(
        g.LineSegments(g.PointsGeometry(verts), g.LineBasicMaterial(color=color))
    )
