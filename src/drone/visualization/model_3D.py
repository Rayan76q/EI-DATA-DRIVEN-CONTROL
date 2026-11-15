"""Drone geometry for the Meshcat viewer: a simple box drone or a glTF model."""

from pathlib import Path

import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np

from .gltf_loader import load_gltf_parts

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TARGET_LENGTH = 0.65  # Same span as the arms of the box drone (modele_drone=1).

# Sketchfab models are Y-up: each rotation sends Y to Z (up) and the front of the model to X.
GLTF_MODELS = {
    2: {
        "file": "drone_model1/scene.gltf",
        "pivot": [0.0, -0.017, -0.010],  # Center of the frame.
        "length": 0.5554,  # Largest dimension of the model.
        "rotation": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],  # Front = +Z of the model.
    },
    3: {
        "file": "drone_model2/scene.gltf",
        "pivot": [0.0, 0.54, -0.21],  # Axis of the rotor nacelles (estimated).
        "length": 3.2656,
        "rotation": [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],  # Front = -Z of the model.
    },
}


def gltf_model_transform(model):
    """Move the pivot to the origin, scale, then orient with X forward and Z up."""
    rotation = np.eye(4)
    rotation[:3, :3] = model["rotation"]
    scale = tf.scale_matrix(TARGET_LENGTH / model["length"])
    centering = tf.translation_matrix(-np.array(model["pivot"]))
    return rotation @ scale @ centering


def creation_modele_drone(visualizer, modele_drone=1):
    """Add the drone under ``visualizer["drone"]``.

    ``modele_drone`` is 1 for a body, two arms and four motor markers, or a
    key of ``GLTF_MODELS`` for a textured model from ``assets/``.
    """
    if modele_drone == 1:
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
        for i, p in enumerate(motor_positions):
            visualizer["drone"][f"motor_{i}"].set_object(
                g.Sphere(0.07), g.MeshLambertMaterial(color=0xFF0000)
            )
            visualizer["drone"][f"motor_{i}"].set_transform(tf.translation_matrix(p))

    elif modele_drone in GLTF_MODELS:
        model = GLTF_MODELS[modele_drone]
        # Separate node: the animation overwrites the transform of visualizer["drone"].
        node = visualizer["drone"]["mesh"]
        for i, (geometry, material) in enumerate(load_gltf_parts(ASSETS_DIR / model["file"])):
            node[f"part_{i}"].set_object(geometry, material)
        node.set_transform(gltf_model_transform(model))
