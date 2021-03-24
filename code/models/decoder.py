# author: Nikola Zubic

import torch
import torch.nn as nn


class Decoder(nn.Module):
    # used to generate point clouds from encoder's hidden vector
    def __init__(self, number_of_point_cloud_points, hidden_dimensions, scale):
        super().__init__()
        self.number_of_point_cloud_points = number_of_point_cloud_points
        self.hidden_dimensions = hidden_dimensions
        self.scale = scale

    def forward(self, hidden_vector):
        """
        Transforms hidden vector to a point cloud.

        :param hidden_vector: given vector
        :return: point cloud and scale factor
        """
        point_cloud = nn.Linear(in_features=self.hidden_dimensions,
                                out_features=self.number_of_point_cloud_points * 3)(hidden_vector)  # decode vector

        point_cloud = point_cloud.view(-1, self.number_of_point_cloud_points, 3)
        point_cloud = torch.tanh(point_cloud) / 2.0

        scaling = None

        if self.scale:
            scaling = nn.Linear(self.hidden_dimensions, 1)(hidden_vector)  # scaling decoder
            scaling = torch.sigmoid(scaling)

        return point_cloud, scaling
