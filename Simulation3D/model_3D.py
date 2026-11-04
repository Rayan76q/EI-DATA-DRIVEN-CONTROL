import meshcat.geometry as g
import meshcat.transformations as tf

def creation_modele_drone(visualizer):
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