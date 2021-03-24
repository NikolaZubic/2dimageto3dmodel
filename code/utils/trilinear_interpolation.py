"""
We use tri-linear interpolation for approximation of function values in one 3D dot (x, y, z) with little prism/cube
in order to get bigger 3D model.

Author: Nikola Zubic
"""

import torch


class TrilinearInterpolation(object):
    def __init__(self, epsilon=1e-6, size=64):
        self.epsilon = epsilon
        self.size = size
        print("Trilinear Interpolation called.")

    def get_point_cloud_object_borders(self, point_cloud):
        """
        :param point_cloud: given point cloud
        :return: point cloud borders
        """

        return (
            (point_cloud < 0.5 - self.epsilon) & (point_cloud > -0.5 + self.epsilon)
        ).all(dim=-1).view(-1)

    def get_grid(self, point_cloud, voxel_size):
        """
        Get grid for a point cloud.
        :param point_cloud: point cloud that will be fitted to a 3D grid
        :param voxel_size: size of a voxel grid
        :return: grid
        """
        grid = (voxel_size - 1) * (point_cloud + 0.5)
        return grid

    def positions_update(self, positions, grid_distance, point_cloud):
        filled_voxels = point_cloud.new(point_cloud.size(0), self.size, self.size, self.size).fill_(0)

        update = grid_distance[positions[0]][..., 0] * grid_distance[positions[1]][..., 1] * \
                 grid_distance[positions[2]][..., 2]

        update = update.view(-1)[self.get_point_cloud_object_borders(point_cloud)]

        indexes_on_grid = self.get_grid(point_cloud=point_cloud, voxel_size=point_cloud.new(3).fill_(self.size))\
            .floor().long()
        indexes_batch = torch.arange(point_cloud.size(0))[:, None, None].repeat(1, point_cloud.size(1), 1)\
            .to(point_cloud.device)

        # indexes format [batch, z, y, x] (z, y, x are for voxel representation)
        indexes = torch.cat([indexes_batch, indexes_on_grid], dim=-1).view(-1, 4)
        indexes = indexes[self.get_point_cloud_object_borders(point_cloud=point_cloud)]

        indexes_shift_right = torch.LongTensor([[0] + positions]).to(point_cloud.device)
        indexes_shift_right = indexes_shift_right.repeat(indexes.size(0), 1)
        indexes_update = indexes + indexes_shift_right

        filled_voxels.index_put_(torch.unbind(indexes_update, dim=1), update, accumulate=True)

        return filled_voxels

    def trilinear_interpolation(self, point_cloud):
        voxels = []
        grid = self.get_grid(point_cloud=point_cloud, voxel_size=point_cloud.new(3).fill_(self.size))

        grid_dist = [1.0 - grid - grid.floor(), grid - grid.floor()]

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    voxels.append(self.positions_update(positions=[i, j, k], grid_distance=grid_dist,
                                                        point_cloud=point_cloud))

        return torch.stack(voxels).sum(dim=0).clamp(min=0, max=1)
