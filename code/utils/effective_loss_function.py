# author: Nikola Zubic

import torch
import torch.nn as nn
from ..camera.coordinate_system_transformation import CameraUtilities
from trilinear_interpolation import TrilinearInterpolation
from smooth_voxels import VoxelsSmooth


class EffectiveLossFunction(nn.Module):
    # Replaces need for differentiable point cloud projection PI(P, c) and generates point cloud without rendering.
    def __init__(self, voxel_size=64, kernel_size=21, smooth_sigma=3.0):
        super(EffectiveLossFunction, self).__init__()
        self.voxel_size = voxel_size
        self.kernel_size = kernel_size
        self.register_buffer("sigma", torch.tensor(smooth_sigma))

    def termination_probs(self, voxels, epsilon=1e-5):
        """
        :param voxels: smoothed voxels
        :param epsilon: ignore factor
        :return:
        """

        """
        Before projecting the resulting volume to a plane, we need to ensure that the signal from the occluded points 
        does not interfere with the foreground points. To this end, we perform occlusion reasoning, similar to 
        Tulsiani et al. [20]. We convert the occupancies o to ray termination probabilities.
        """

        # The occupancy function of the point cloud is a clipped sum of the individual per-point functions
        per_point_functions = voxels.permute(1, 0, 2, 3)
        occupancy_function = per_point_functions.clamp(epsilon, 1.0 - epsilon)

        x = torch.log(1 - occupancy_function)
        x_prim = torch.log(occupancy_function)

        ray_termination_probs = torch.cumsum(x, dim=0)

        zeros_matrix = voxels.new(1, occupancy_function.size(1), occupancy_function.size(2),
                                  occupancy_function.size(3)).fill_(epsilon)

        """
        Intuitively, a cell has high termination probability if its occupancy value is high and all previous occupancy 
        values are low. The additional background cell serves to ensure that the termination probabilities sum to 1.
        """

        # Concatenates the given sequence of seq tensors in the given dimension
        r1 = torch.cat([zeros_matrix, ray_termination_probs], dim=0)

        # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        r2 = torch.cat([x_prim, zeros_matrix], dim=0)

        project_volume_to_the_plane = r1 + r2

        return torch.exp(project_volume_to_the_plane).permute(1, 0, 2, 3)

    def forward(self, point_cloud, rotation, scale=None):
        """
        Projection based loss.

        :param point_cloud: point cloud of interest
        :param rotation: is rotation specified
        :param scale: if not None will scale the object
        :return: projection
        """
        camera_utilities = CameraUtilities()
        point_cloud = camera_utilities.transformation_3d_coord_to_camera_coord(point_cloud=point_cloud,
                                                                               rotation=rotation, field_of_view=1.875,
                                                                               camera_view_distance=2.0)

        interpolation = TrilinearInterpolation()

        voxels = interpolation.trilinear_interpolation(point_cloud=point_cloud)
        voxels_smoothing = VoxelsSmooth()

        smoothed_voxels = voxels_smoothing.smooth(voxels=voxels, kernels=(), scale=scale)

        probs = self.termination_probs(smoothed_voxels)

        return probs[:, :-1].sum(1).flip(1)
