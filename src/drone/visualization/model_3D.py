"""Simple drone geometry for the optional Meshcat viewer."""

import meshcat.geometry as g
import meshcat.transformations as tf


def creation_modele_drone(visualizer):
    """Build a body, two arms and four motor markers under the drone node."""
    visualizer["drone"]["body"].set_object(
        g.Box([0.25, 0.20, 0.08]), g.MeshLambertMaterial(color=0x333333)
    )
    visualizer["drone"]["arm_x"].set_object(
        g.Box([0.65, 0.045, 0.035]), g.MeshLambertMaterial(color=0x222222)
    )
    visualizer["drone"]["arm_y"].set_object(
        g.Box([0.045, 0.65, 0.035]), g.MeshLambertMaterial(color=0x222222)
    )
    motor_positions = [
        [0.30, 0.30, 0],
        [0.30, -0.30, 0],
        [-0.30, 0.30, 0],
        [-0.30, -0.30, 0],
    ]
    for index, position in enumerate(motor_positions):
        node = visualizer["drone"][f"motor_{index}"]
        node.set_object(g.Sphere(0.07), g.MeshLambertMaterial(color=0xFF0000))
        node.set_transform(tf.translation_matrix(position))
