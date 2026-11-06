import numpy as np
from scipy.spatial.transform import Rotation
import meshcat.animation as animation
import meshcat.geometry as g
import meshcat.transformations as tf

def drone_transform(x, y, z, phi, theta, psi):
    T = np.zeros((4, 4))

    # Rotation : roulis, tangage, lacet
    R = Rotation.from_euler(
        "ZYX",
        [psi, theta, phi]
    ).as_matrix()

    # Construction de la transformation
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    T[3,3] = 1

    return T


def animate_episode(data, visualizer, fps = 30, color_change_speed = 80, traj_radius = 0.01):#speed in simulation
    t = data[:,0]
    x = data[:,1]
    y = data[:,2]
    z = data[:,3]
    phi = data[:,4]
    theta = data[:,5]
    psi = data[:,6]
    
    anim = animation.Animation(default_framerate=fps)
    
    def time_to_fps(t):
        return fps * t
    
    # draw trajectory 
    for i in range(len(t) - 1):
        
        material = g.MeshLambertMaterial(color=calculate_color(t[i], color_change_speed))
        T, L = segment_transform([x[i], y[i], z[i]], [x[i+1], y[i+1], z[i+1]])
        if T is None:                         # drone did not move: skip zero-length segment
            continue
        segment_path = visualizer["world/trajectory"][f"segment_{i}"]
        segment_path.set_object(g.Cylinder(L, traj_radius), material)
        segment_path.set_transform(T)
        segment_path.set_property("visible", False)
        
        with anim.at_frame(segment_path, 0) as frame:
            frame.set_property("visible", "bool", False)
        with anim.at_frame(segment_path, time_to_fps(t[i+1])) as frame:
            frame.set_property("visible", "bool", True)            
    
    #animate drone 
    for i, timestamp in enumerate(t):
        frame_nb = time_to_fps(timestamp)
        with anim.at_frame(visualizer["drone"], frame_nb) as frame:
            #frame.set_transform(drone_transform(x[i],y[i],z[i],phi[i],theta[i],psi[i]))
            frame.set_transform(tf.euler_matrix(phi[i], theta[i], psi[i], "sxyz"))  # attitude only
        with anim.at_frame(visualizer["world"], frame_nb) as frame:
            frame.set_transform(tf.translation_matrix([-x[i], -y[i], -z[i]]))

            
    visualizer.set_animation(anim)
    
def calculate_color(t, color_change_speed):
    hue = (t * color_change_speed)%360
    return hsv_to_hex(hue, 100, 100)

def hsv_to_hex(h, s, v):
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
    """4x4 transform placing a y-aligned Cylinder of height L between p0 and p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-9:
        return None, 0.0
    d = d / L
    y = np.array([0.0, 1.0, 0.0])           # Three.js cylinders point along y
    axis = np.cross(y, d)
    s = np.linalg.norm(axis)
    if s < 1e-9:                              # d parallel or anti-parallel to y
        R = np.eye(4) if d[1] > 0 else tf.rotation_matrix(np.pi, [1, 0, 0])
    else:
        R = tf.rotation_matrix(np.arctan2(s, np.dot(y, d)), axis / s)
    return tf.translation_matrix((p0 + p1) / 2) @ R, L

