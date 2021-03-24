# author: Nikola Zubic

import torch
import torch.nn as nn
from basic_blocks import BasicBlocks


class PoseDecoder(nn.Module):
    """
    In pose part, hidden dimensions are bigger than in the architecture by Insafutdinov and Dosovitskiy.
    Idea of pose decoder:
        * training mode => returns Ensemble of pose regressors (specified by a number parameter) for each example
        where an additional pose is student's prediction. So, for every example we have certain number of pose
        candidates. For every view we can produce point clouds and get projections.
        * test mode => returns one pose for each example in batch that is considered as the best.
    """
    def __init__(self, input_dimensions, hidden_dimensions, number_of_pose_candidates):
        super().__init__()

        """
        The idea is that each of the predictors learns to specialize on a subset of poses and together they cover
        the whole range of possible values. No special measures are needed to ensure this specialization: it
        emerges naturally as a result of random weight initialization if the network architecture is appropriate.
        """

        basic_blocks = BasicBlocks()

        self.ensemble_of_pose_regressors = nn.Sequential(
            nn.Linear(input_dimensions, hidden_dimensions),
            nn.ReLU(True)
        )

        self.predictors = nn.ModuleList(
            [basic_blocks.pose_prediction(hidden_dimensions) for _ in range(number_of_pose_candidates)]
        )

        # Namely, the different pose predictors need to have several (at least 3, in our experience) non-shared layers
        self.student_predictor = nn.Sequential(
            nn.Linear(input_dimensions, hidden_dimensions),
            nn.ReLU(True),
            basic_blocks.pose_prediction(hidden_dimensions)
        )

    def forward(self, hidden_vector):
        """
        Transformation of hidden vector to rotation quaternions.
        The loss for training of the student  is computed as an angular difference between two rotations represented by
        quaternions: L(q_1, q_2) = 1 - Re(q_1 * q_2^(-1) / || q_1 * q_2^(-1) ||), where Re denotes real part of the
        quaternion. Standard MSE loss performs poorly when regressing rotation.

        :param hidden_vector: given hidden vector
        :return: rotation quaternions
        """
        student_rotation_quaternions = self.student_predictor(hidden_vector)

        if not self.training:
            return student_rotation_quaternions

        shared_part = self.ensemble_of_pose_regressors(hidden_vector)

        quaternions = [quaternion(shared_part) for quaternion in self.predictors]

        ensemble_quaternions = torch.cat(quaternions, dim=-1).view(-1, 4)

        # The camera pose is predicted as a quaternion
        return torch.cat([ensemble_quaternions, student_rotation_quaternions], dim=0)
