# author: Nikola Zubic

import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from ..utils.dropout import PointCloudDropOut
from ..utils.effective_loss_function import EffectiveLossFunction
from ..utils.batch_repetition import repeat_tensor_for_each_element_in_batch
import torch.nn.functional as F


class SupervisedPart(nn.Module):
    """
    This model will use camera supervision to predict point clouds with different camera viewpoints, these projections
    are used with supervised loss to induce the conclusion.
    """
    def __init__(self, image_size, hidden_dimensions, number_of_point_cloud_points, voxel_size, smoothness_factor,
                 predict_scale, keep_probability):
        super().__init__()

        self.encoder = Encoder(image_size=image_size)
        self.point_cloud_decoder = Decoder(number_of_point_cloud_points=number_of_point_cloud_points,
                                           hidden_dimensions=hidden_dimensions, scale=predict_scale)

        self.point_cloud_drop_out = PointCloudDropOut(p=keep_probability)

        self.effective_loss_function = EffectiveLossFunction(voxel_size=voxel_size, smooth_sigma=smoothness_factor)

        self.encoder.apply(self.kaiming_initialization)
        self.decoder.apply(self.kaiming_initialization)

    def kaiming_initialization(self, m):
        """
        https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138

        :param m: apply on certain architecture
        :return: initializes weights with Kaiming Initialization
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight.data, a=0)

    def forward(self, images, poses):
        """
        Generate new view of images from cameras using effective loss function.

        :param images: all images in a batch
        :param poses: given poses attached to images
        :return: new projection views
        """
        batch_size = images.size(0)

        number_of_views = poses.size(0) // batch_size

        encode_images_hidden_vector = self.encoder(images)

        point_cloud, scaling = self.point_cloud_decoder(encode_images_hidden_vector)

        point_cloud = repeat_tensor_for_each_element_in_batch(torch_tensor=self.point_cloud_drop_out(point_cloud),
                                                              n=number_of_views)
        scaling = repeat_tensor_for_each_element_in_batch(torch_tensor=scaling, n=number_of_views)

        projection = self.effective_loss_function.forward(point_cloud=point_cloud, rotation=poses, scale=scaling)

        return projection


class SupervisedLoss(nn.Module):
    def forward(self, projection, masks, **kwargs):
        masks = F.interpolate(input=masks.unsqueeze(0), scale_factor=1/2, mode="bilinear", align_corners=True).squeeze()
        return dict(full_loss=F.mse_loss(input=projection, target=masks, reduce=None,
                                         reduction="sum") / (2 * projection.size(0)))
