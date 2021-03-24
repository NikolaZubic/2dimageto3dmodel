"""

author: Nikola Zubic
"""

from math import sqrt, pow, acos, pi, asin
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch


def scale_to_n(axis, n):
    return axis / n


def blender_camera_position_to_torch_tensor_quaternion(blender_camera_info):
    x, y, z = blender_camera_info[0]

    # distance from camera
    d = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2))

    x, y, z = scale_to_n(x, d), scale_to_n(y, d), scale_to_n(z, d)

    d_2D = sqrt(pow(x, 2) + pow(y, 2))

    x2D, y2D = scale_to_n(x, d_2D), scale_to_n(y, d_2D)

    # z axis
    yaw = acos(x2D)

    if y2D > 0:
        yaw = 2 * pi - yaw

    """
    Yaw, pitch and roll is a way of describing the rotation of the camera in 3D. There is other ways like quaternions 
    but this is the simplest. Yaw, pitch and roll is the name of how much we should rotate around each axis.
    Think about yourself as the camera right now. Look around a bit. Yaw is the angle when moving the head 
    left <=> right (rotation around Y-axis). Pitch is up and down (rotation around X-axis). Roll, which we usually 
    don't experience is when you tilt your head (rotation around Z-axis).
    """
    roll = 0
    pitch = asin(z)
    yaw = yaw + pi

    # Initialize from Euler angles
    quaternion = R.from_euler(
        seq="yzx",
        angles=[yaw, pitch, roll]  # Euler angles specified in radians
    ).as_quat()

    # form matrix: scalar part, vector part
    quaternion = np.r_[quaternion[-1], quaternion[:-1]]

    return torch.tensor(quaternion.astype(
        dtype=np.float32
    ))
