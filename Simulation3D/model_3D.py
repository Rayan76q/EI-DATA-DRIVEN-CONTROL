from pathlib import Path

import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf

from .gltf_loader import load_gltf_parts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_LENGTH = 0.65  # same span as the arms of the box drone (modele_drone = 1)

# Les modèles Sketchfab sont en Y-up : chaque rotation envoie Y sur Z (haut) et l'avant du modèle sur X.
GLTF_MODELS = {
    2: {
        "file": "drone_model1/scene.gltf",
        "pivot": [0.0, -0.017, -0.010],  # centre du châssis
        "length": 0.5554,                # plus grande dimension du modèle
        "rotation": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],  # avant = +Z du modèle
    },
    3: {
        "file": "drone_model2/scene.gltf",
        "pivot": [0.0, 0.54, -0.21],     # axe des nacelles (estimé)
        "length": 3.2656,
        "rotation": [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],  # avant = -Z du modèle
    },
}


def gltf_model_transform(model):
    """Place le modèle : pivot à l'origine, mise à l'échelle, puis orientation X-avant, Z-haut."""
    rotation = np.eye(4)
    rotation[:3, :3] = model["rotation"]
    scale = tf.scale_matrix(TARGET_LENGTH / model["length"])
    centering = tf.translation_matrix(-np.array(model["pivot"]))
    return rotation @ scale @ centering


def creation_modele_drone(visualizer, modele_drone = 1):
    if modele_drone == 1:
        visualizer["drone"]["body"].set_object(
            g.Box([0.25, 0.20, 0.08]),
            g.MeshLambertMaterial(color=0x333333)
        )
        visualizer["drone"]["arm_x"].set_object(
            g.Box([0.65, 0.045, 0.035]),
            g.MeshLambertMaterial(color=0x222222)
        )
        visualizer["drone"]["arm_y"].set_object(
            g.Box([0.045, 0.65, 0.035]),
            g.MeshLambertMaterial(color=0x222222)
        )
        motor_positions = [
            [ 0.30,  0.30, 0],
            [ 0.30, -0.30, 0],
            [-0.30,  0.30, 0],
            [-0.30, -0.30, 0]
        ]
        for i, p in enumerate(motor_positions):
            visualizer["drone"][f"motor_{i}"].set_object(
                g.Sphere(0.07),
                g.MeshLambertMaterial(color=0xff0000)
            )
            visualizer["drone"][f"motor_{i}"].set_transform(
                tf.translation_matrix(p)
            )
    
    elif modele_drone in GLTF_MODELS:
        model = GLTF_MODELS[modele_drone]
        # Noeud séparé : l'animation écrase la transformation de visualizer["drone"]
        node = visualizer["drone"]["mesh"]
        for i, (geometry, material) in enumerate(load_gltf_parts(PROJECT_ROOT / model["file"])):
            node[f"part_{i}"].set_object(geometry, material)
        node.set_transform(gltf_model_transform(model))