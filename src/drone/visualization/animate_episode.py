"""Animate a pose sequence in Meshcat while keeping the drone centered."""

import meshcat.animation as animation
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from scipy.spatial.transform import Rotation


def drone_transform(x, y, z, phi, theta, psi):
    """Build the 4x4 pose of a drone at (x, y, z) with roll, pitch and yaw."""
    T = np.zeros((4, 4))
    T[:3, :3] = Rotation.from_euler("ZYX", [psi, theta, phi]).as_matrix()
    T[:3, 3] = [x, y, z]
    T[3, 3] = 1
    return T


def animate_episode(data, visualizer, fps=30, color_change_speed=80, traj_radius=0.01):
    """Animate rows [t,x,y,z,phi,theta,psi] and reveal the path as it is flown.

    The drone only changes attitude. The world moves the opposite way, so the
    drone stays centered while its path and the grid move past it. The path
    hue turns by ``color_change_speed`` degrees per second of flight.
    """
    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    phi = data[:, 4]
    theta = data[:, 5]
    psi = data[:, 6]

    anim = animation.Animation(default_framerate=fps)

    # Path: one cylinder per step, hidden until the drone reaches its end.
    for i in range(len(t) - 1):
        material = g.MeshLambertMaterial(color=calculate_color(t[i], color_change_speed))
        T, L = segment_transform([x[i], y[i], z[i]], [x[i + 1], y[i + 1], z[i + 1]])
        if T is None:  # The drone did not move: skip the zero-length segment.
            continue
        segment_path = visualizer["world/trajectory"][f"segment_{i}"]
        segment_path.set_object(g.Cylinder(L, traj_radius), material)
        segment_path.set_transform(T)
        segment_path.set_property("visible", False)
        with anim.at_frame(segment_path, 0) as frame:
            frame.set_property("visible", "bool", False)
        with anim.at_frame(segment_path, fps * t[i + 1]) as frame:
            frame.set_property("visible", "bool", True)

    # Drone: attitude only, while the world is translated by minus its position.
    for i, timestamp in enumerate(t):
        frame_nb = fps * timestamp
        with anim.at_frame(visualizer["drone"], frame_nb) as frame:
            frame.set_transform(tf.euler_matrix(phi[i], theta[i], psi[i], "sxyz"))
        with anim.at_frame(visualizer["world"], frame_nb) as frame:
            frame.set_transform(tf.translation_matrix([-x[i], -y[i], -z[i]]))

    visualizer.set_animation(anim)


def calculate_color(t, color_change_speed):
    """Return the path color at time t; the hue cycles over 360 degrees."""
    hue = (t * color_change_speed) % 360
    return hsv_to_hex(hue, 100, 100)


def hsv_to_hex(h, s, v):
    """Convert hue in degrees and saturation/value in percent to 0xRRGGBB."""
    s /= 100
    v /= 100

    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = round((r + m) * 255)
    g = round((g + m) * 255)
    b = round((b + m) * 255)
    return (r << 16) | (g << 8) | b


def segment_transform(p0, p1):
    """Return the pose and length of a y-aligned cylinder from p0 to p1.

    The pose is None when both points coincide.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-9:
        return None, 0.0
    d = d / L
    y = np.array([0.0, 1.0, 0.0])  # Three.js cylinders point along y.
    axis = np.cross(y, d)
    s = np.linalg.norm(axis)
    if s < 1e-9:  # d is parallel or opposite to y.
        R = np.eye(4) if d[1] > 0 else tf.rotation_matrix(np.pi, [1, 0, 0])
    else:
        R = tf.rotation_matrix(np.arctan2(s, np.dot(y, d)), axis / s)
    return tf.translation_matrix((p0 + p1) / 2) @ R, L
