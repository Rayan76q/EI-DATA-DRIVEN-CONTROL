"""Create the optional Meshcat scene and its world reference geometry."""

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np


def initialize_visualizer():
    """Start Meshcat, open its browser view and print the local viewer URL."""
    visualizer = meshcat.Visualizer()
    visualizer.open()
    print(visualizer.url())
    return visualizer


def set_axes(visualizer, length=5.0, radius=0.5):
    """Draw colored world axes that move with the centered scene."""
    visualizer["/Axes"].set_property("visible", False)
    rotations = {
        "x": (0xFF0000, tf.rotation_matrix(-np.pi / 2, [0, 0, 1])),
        "y": (0x00FF00, np.eye(4)),
        "z": (0x0000FF, tf.rotation_matrix(np.pi / 2, [1, 0, 0])),
    }
    shift = tf.translation_matrix([0, length / 2, 0])
    for name, (color, rotation) in rotations.items():
        node = visualizer[f"world/world_axes/{name}"]
        node.set_object(g.Cylinder(length, radius), g.MeshLambertMaterial(color=color))
        node.set_transform(rotation @ shift)


def add_moving_grid(visualizer, spacing=1.0, n=30, color=0x888888):
    """Draw a ground grid in the world frame instead of Meshcat's fixed grid."""
    visualizer["/Grid"].set_property("visible", False)
    extent = n * spacing
    points = []
    for offset in np.arange(-n, n + 1) * spacing:
        points.extend([
            [offset, -extent, 0], [offset, extent, 0],
            [-extent, offset, 0], [extent, offset, 0],
        ])
    vertices = np.array(points, dtype=np.float32).T
    visualizer["world/grid"].set_object(
        g.LineSegments(g.PointsGeometry(vertices), g.LineBasicMaterial(color=color))
    )
