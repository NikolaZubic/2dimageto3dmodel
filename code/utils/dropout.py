# author: Nikola Zubic

import torch
import torch.nn as nn
from batch_repetition import repeat_tensor_for_each_element_in_batch
import math


class PointCloudDropOut(nn.Module):
    def __init__(self, p):
        """
        Point Cloud Drop-out class which drops some portions of point cloud and still retains important structure.
        Keep only points with p probability in each point cloud of a batch.

        :param p: probability for keeping point(s)
        """
        super().__init__()
        self.p = p

    def forward(self, point_cloud):
        """
        Forward drop-out propagation.

        :param point_cloud: point cloud of interest
        :return: different points for each example in one batch
        """
        batch_size = point_cloud.size(0)
        number_of_point_cloud_points = point_cloud.size(1)

        batch_indexes = repeat_tensor_for_each_element_in_batch(torch_tensor=torch.arange(batch_size),
                                                                n=math.ceil(number_of_point_cloud_points * self.p))

        points_indexes = torch.cat([torch.randperm(number_of_point_cloud_points)
                                    [:math.ceil(number_of_point_cloud_points * self.p)] for _ in range(batch_size)])

        return point_cloud[batch_indexes, points_indexes].view(batch_size,
                                                               math.ceil(number_of_point_cloud_points * self.p), -1)
