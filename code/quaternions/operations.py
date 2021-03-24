"""
https://www.haroldserrano.com/blog/quaternions-in-computer-graphics#:~:text=Quaternions%20are%20mainly%20used%20in,sequentially%20as%20matrix%20rotation%20allows.&text=Matrix%20rotations%20suffer%20from%20what%20is%20known%20as%20Gimbal%20Lock.

author: Nikola Zubic
"""

import torch
from math import pow


class QuaternionOperations(object):
    def __init__(self):
        print("Quaternion Operations called.")

    def quaternion_addition(self, q1, q2):
        """
        Function for addition of two quaternions.
        :param q1: first quaternion
        :param q2: second quaternion
        :return: result of the addition q1 + q2
        """

        """
        Unpack these quaternions.
        
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(torch.unbind(tensor)) => (tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))
        """
        a_scalar, a_vecx, a_vecy, a_vecz = torch.unbind(q1,
                                                        dim=-1)
        b_scalar, b_vecx, b_vecy, b_vecz = torch.unbind(q2,
                                                        dim=-1)

        r_scalar = a_scalar + b_scalar
        r_vecx = a_vecx + b_vecx
        r_vecy = a_vecy + b_vecy
        r_vecz = a_vecz + b_vecz

        return torch.stack(
            [r_scalar, r_vecx, r_vecy, r_vecz],
            dim=-1
        )

    def quaternion_subtraction(self, q1, q2):
        """
        Function for subtraction of two quaternions.
        :param q1: first quaternion
        :param q2: second quaternion
        :return: result of the subtraction q1 - q2
        """

        # Unpack these quaternions
        a_scalar, a_vecx, a_vecy, a_vecz = torch.unbind(q1,
                                                        dim=-1)
        b_scalar, b_vecx, b_vecy, b_vecz = torch.unbind(q2,
                                                        dim=-1)

        r_scalar = a_scalar - b_scalar
        r_vecx = a_vecx - b_vecx
        r_vecy = a_vecy - b_vecy
        r_vecz = a_vecz - b_vecz

        return torch.stack(
            [r_scalar, r_vecx, r_vecy, r_vecz],
            dim=-1
        )

    def quaternion_multiplication(self, q1, q2):
        """
        Function for multiplication of two quaternions.
        :param q1: first quaternion
        :param q2: second quaternion
        :return: result of the multiplication q1 * q2
        """

        # Unpack these quaternions
        a_scalar, a_vecx, a_vecy, a_vecz = torch.unbind(q1,
                                                        dim=-1)
        b_scalar, b_vecx, b_vecy, b_vecz = torch.unbind(q2,
                                                        dim=-1)

        r_scalar = a_scalar * b_scalar - a_vecx * b_vecx - a_vecy * b_vecy - a_vecz * b_vecz
        r_vecx = a_scalar * b_vecx + a_vecx * b_scalar + a_vecy * b_vecz - a_vecz * b_vecy
        r_vecy = a_scalar * b_vecy + a_vecy * b_scalar + a_vecz * b_vecx - a_vecx * b_vecz
        r_vecz = a_scalar * b_vecz + a_vecz * b_scalar + a_vecx * b_vecy - a_vecy * b_vecx

        """
        a = torch.randn([2, 3, 4])
        b = torch.randn([2, 3, 4])
        print(a) # 2 matrices of size 3 x 4
        print(b) # 2 matrices of size 3 x 4
        print(torch.stack([a, b])) # 4 matrices of size 3 x 4, first a, then b
        """
        return torch.stack(
            [r_scalar, r_vecx, r_vecy, r_vecz],
            dim=-1
        )

    def quaternion_square(self, q):
        """
        Function for squaring the quaternion.
        :param q: qauternion to be squared
        :return: result of the squaring q*q
        """

        # Unpack the quaternion
        a_scalar, a_vecx, a_vecy, a_vecz = torch.unbind(q,
                                                        dim=-1)

        r_scalar = pow(a_scalar, 2) - pow(a_vecx, 2) - pow(a_vecy, 2) - pow(a_vecz, 2)
        r_vecx = 2 * a_scalar * a_vecx
        r_vecy = 2 * a_scalar * a_vecy
        r_vecz = 2 * a_scalar * a_vecz

        return torch.stack(
            [r_scalar, r_vecx, r_vecy, r_vecz],
            dim=-1
        )

    def quaternion_conjugate(self, q):
        """
        Function for computing the inverse of a quaternion.
        :param q: quaternion to be inverted
        :return: result of the inversion of q
        """

        """
        in-place operation is an operation that changes directly the content of a given Tensor without making a copy.
        ALL operations on the tensor that operate in-place on it will have an _ postfix.
        """
        q_star = q.new(4).fill_(-1)

        # leave the scalar unchanged and change signs of i, j, k number parts
        q_star[0] = 1.0

        return q * q_star
