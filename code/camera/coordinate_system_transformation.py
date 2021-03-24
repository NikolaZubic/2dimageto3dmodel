"""
Traditional pipeline in computer graphics renders images from the viewpoint of a virtual pin-hole
camera by using the very frequent perspective projection.

View direction is initially set along the negative z-axis in camera coordinate system.
So, 3D content which we define needs a transformation from ordinary 3D coordinate system in
camera coordinate system before we do the operations on it.

Author: Nikola Zubic
"""

from ..quaternions.points_quaternions import PointsQuaternionsRotator
import torch


class CameraUtilities(object):
    def __init__(self):
        print("Camera Utilities called.")

    def transformation_3d_coord_to_camera_coord(self, point_cloud, rotation, field_of_view, camera_view_distance):
        points_quaternions_rotator = PointsQuaternionsRotator()
        point_cloud = points_quaternions_rotator.rotate_points(point_cloud, rotation,
                                                               inverse_rotation_direction=False)

        z, y, x = torch.unbind(point_cloud, dim=2)

        """
        The Focal Length / Field of View controls the amount of zoom, i.e. the amount of the scene which is 
        visible all at once. Longer focal lengths result in a smaller FOV (more zoom), while short focal 
        lengths allow you to see more of the scene at once (larger FOV, less zoom).
        """

        x = x * field_of_view / (z + camera_view_distance)
        y = y * field_of_view / (z + camera_view_distance)

        return torch.stack(
            [z, y, x],
            dim=2  # z is same with different angles and x, y positions
        )
