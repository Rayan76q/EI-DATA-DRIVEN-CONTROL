"""Animate poses while keeping the drone centered in the Meshcat view."""

import meshcat.animation as animation
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from scipy.spatial.transform import Rotation


def drone_transform(x, y, z, phi, theta, psi):
    """Build a homogeneous pose matrix from position and roll, pitch, yaw."""
    transform = np.zeros((4, 4))
    transform[:3, :3] = Rotation.from_euler("ZYX", [psi, theta, phi]).as_matrix()
    transform[:3, 3] = [x, y, z]
    transform[3, 3] = 1
    return transform


def animate_episode(data, visualizer, fps=30, color_change_speed=80, traj_radius=0.01):
    """Animate rows [t,x,y,z,phi,theta,psi] and reveal the path over time.

    The drone changes attitude in place. Translating the world in the opposite
    direction keeps the drone centered while its path and ground move past it.
    """
    time, x, y, z, phi, theta, psi = data.T
    animation_frames = animation.Animation(default_framerate=fps)

    for index in range(len(time) - 1):
        material = g.MeshLambertMaterial(color=calculate_color(time[index], color_change_speed))
        transform, length = segment_transform(
            [x[index], y[index], z[index]],
            [x[index + 1], y[index + 1], z[index + 1]],
        )
        if transform is None:
            continue  # A stationary pose needs no path segment.
        segment = visualizer["world/trajectory"][f"segment_{index}"]
        segment.set_object(g.Cylinder(length, traj_radius), material)
        segment.set_transform(transform)
        segment.set_property("visible", False)
        with animation_frames.at_frame(segment, 0) as frame:
            frame.set_property("visible", "bool", False)
        with animation_frames.at_frame(segment, fps * time[index + 1]) as frame:
            frame.set_property("visible", "bool", True)

    for index, timestamp in enumerate(time):
        frame_number = fps * timestamp
        with animation_frames.at_frame(visualizer["drone"], frame_number) as frame:
            frame.set_transform(tf.euler_matrix(phi[index], theta[index], psi[index], "sxyz"))
        with animation_frames.at_frame(visualizer["world"], frame_number) as frame:
            frame.set_transform(tf.translation_matrix([-x[index], -y[index], -z[index]]))
    visualizer.set_animation(animation_frames)


def calculate_color(time, color_change_speed):
    """Cycle trajectory hue in degrees per second."""
    hue = (time * color_change_speed) % 360
    return hsv_to_hex(hue, 100, 100)


def hsv_to_hex(h, s, v):
    """Convert hue in degrees and saturation/value percentages to #RRGGBB."""
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
    return f"#{r:02X}{g:02X}{b:02X}"


def segment_transform(p0, p1):
    """Return the pose and length of a y-aligned cylinder between two points."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return None, 0.0

    direction = direction / length
    cylinder_axis = np.array([0.0, 1.0, 0.0])
    axis = np.cross(cylinder_axis, direction)
    sine = np.linalg.norm(axis)
    if sine < 1e-9:
        # Parallel directions need no rotation; opposite directions need a half turn.
        rotation = np.eye(4) if direction[1] > 0 else tf.rotation_matrix(np.pi, [1, 0, 0])
    else:
        angle = np.arctan2(sine, np.dot(cylinder_axis, direction))
        rotation = tf.rotation_matrix(angle, axis / sine)
    return tf.translation_matrix((p0 + p1) / 2) @ rotation, length
