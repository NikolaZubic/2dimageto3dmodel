"""

author: Nikola Zubic
"""

import torch
import torch.nn.functional as F
from .operations import QuaternionOperations


class PointsQuaternionsConverter(object):
    def __init__(self):
        print("Points Quaternions Converter called.")

    @staticmethod
    def points_to_quaternions(xyz_triplet):
        """
        Convert (x, y, z) to quaternion representation.
        :param xyz_triplet: triplet to be converted
        :return: quaternion formed from triplet
        """

        assert not len(xyz_triplet) != 3
        assert not xyz_triplet.size(-1) != 3

        """
        'pad' parameter defines how much padding to give pad = (x, y, z, u), where x is the width for left padding, 
        y is for right padding, z is for upper padding and u is for lower padding.
        For example: if we have x, y, z triplet: tensor([[0.5805, 0.8800, 0.1378]]).
        Padding (1, 0, 0, 0) will give => tensor([[0.0000, 0.5805, 0.8800, 0.1378]]).
        With this we basically have quaternion (scalar + vector).
        """
        return F.pad(input=xyz_triplet, pad=(1, 0, 0, 0))


class PointsQuaternionsRotator(object):
    def __init__(self):
        print("Points Quaternions Rotator called.")

    @staticmethod
    def rotate_points(xyz_triplet, q, inverse_rotation_direction):
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

        """
        Rotate point triplet (x, y, z) with quaternion q.
        :param xyz_triplet: triplet for rotation
        :param q: quaternion for rotating the point
        :param inverse_rotation_direction: rotation direction
        :return: rotated point
        """

        q = F.normalize(
            q,
            dim=-1
        )

        # for quaternion tensor([[scalar, x, y, z]]), get tensor([[[scalar, x, y, z]]])
        q = q[:, None, :]

        quaternion_operations = QuaternionOperations()
        q_star = quaternion_operations.quaternion_conjugate(q)

        points_quaternions_converter = PointsQuaternionsConverter()
        xyz_triplet = points_quaternions_converter.points_to_quaternions(xyz_triplet)

        if inverse_rotation_direction:
            new_point = quaternion_operations.quaternion_multiplication(
                quaternion_operations.quaternion_multiplication(q_star, xyz_triplet),
                q
            )
        else:
            new_point = quaternion_operations.quaternion_multiplication(
                quaternion_operations.quaternion_multiplication(q, xyz_triplet),
                q_star
            )

        if len(new_point.shape) == 2:
            new_point = new_point.unsqueeze(0)  # if after multiplication we have [[]], create [[[]]]

        # 1:4 since we only return tensor([[[x, y, z]]])
        return new_point[:, :, 1:4]
