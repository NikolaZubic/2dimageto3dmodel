# author: Nikola Zubic

import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from pose_decoder import PoseDecoder
from ..utils.dropout import PointCloudDropOut
from ..utils.effective_loss_function import EffectiveLossFunction
from ..utils.batch_repetition import repeat_tensor_for_each_element_in_batch
import torch.nn.functional as F
from ..quaternions.operations import QuaternionOperations


class UnsupervisedPart(nn.Module):
    #  Unsupervised model that uses ensemble of pose predictors and effective loss function

    def __init__(self, image_size=128, voxel_size=64, z_dimension=1024, pose_dimensions=128,
                 number_of_point_cloud_points=8000, number_of_pose_predictor_candidates=4, number_of_views=5):
        """

        :param image_size: image size
        :param voxel_size: voxel size (after tri-linear interpolation)
        :param z_dimension: dimension used for encoder-decoders
        :param pose_dimensions: dimension used for pose decoder
        :param number_of_point_cloud_points: number of point cloud points used when decoding it
        :param number_of_pose_predictor_candidates: number of candidates from which we 'll use the best one
        :param number_of_views: number of image views
        """
        super().__init__()

        self.encoder = Encoder(image_size=image_size)
        self.decoder = Decoder(number_of_point_cloud_points=number_of_point_cloud_points, hidden_dimensions=z_dimension,
                               scale=True)
        self.point_cloud_drop_out = PointCloudDropOut(p=0.07)
        self.effective_loss_function = EffectiveLossFunction(voxel_size=voxel_size)
        self.pose_decoder = PoseDecoder(input_dimensions=z_dimension, hidden_dimensions=pose_dimensions,
                                        number_of_pose_candidates=number_of_pose_predictor_candidates)

        self.number_of_views = number_of_views
        self.number_of_pose_predictor_candidates = number_of_pose_predictor_candidates

        self.encoder.apply(self.kaiming_initialization)
        self.decoder.apply(self.kaiming_initialization)
        self.pose_decoder.apply(self.kaiming_initialization)

    @staticmethod
    def kaiming_initialization(architecture):
        # Kaiming initialization for encoder, decoder and pose decoder
        if isinstance(architecture, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(architecture.weight.data, a=0)

    def forward(self, images, poses):
        """

        :param images: all images in a batch
        :param poses: given poses attached to images
        :return: new projection views, ensemble and student poses
        """
        encoder_image_features = self.encoder.forward(images)
        encoder_pose_features = self.encoder.forward(poses)

        point_cloud, scaling = self.decoder.forward(hidden_vector=encoder_image_features)
        poses = self.pose_decoder.forward(hidden_vector=encoder_pose_features)

        # do not create ensemble of pose predictors if we aren't training
        if not self.training:
            point_clouds = repeat_tensor_for_each_element_in_batch(torch_tensor=self.point_cloud_drop_out.
                                                                   forward(point_cloud=point_cloud),
                                                                   n=self.number_of_views)
            scalings = repeat_tensor_for_each_element_in_batch(torch_tensor=scaling, n=self.number_of_views)
            projection = self.effective_loss_function.forward(point_cloud=point_clouds, rotation=poses, scale=scalings)

            return projection, poses

        batch_size = images.size(0) * self.number_of_views
        ensemble_poses, student_poses = poses[:-batch_size], poses[-batch_size:]

        point_clouds = repeat_tensor_for_each_element_in_batch(torch_tensor=self.point_cloud_drop_out.
                                                               forward(point_cloud=point_cloud),
                                                               n=self.number_of_pose_predictor_candidates * self.
                                                               number_of_views)
        scalings = repeat_tensor_for_each_element_in_batch(torch_tensor=scaling, n=self.
                                                           number_of_pose_predictor_candidates * self.number_of_views)
        projection = self.effective_loss_function.forward(point_cloud=point_clouds, rotation=poses, scale=scalings)

        return projection, ensemble_poses, student_poses


class UnsupervisedLoss(nn.Module):
    # Combines projection effective losses for ensemble and student loss
    def __init__(self, number_of_pose_predictor_candidates=4, student_weight=20.00):
        super().__init__()
        self.student_weight = student_weight
        self.number_of_pose_predictor_candidates = number_of_pose_predictor_candidates
        self.minimum_indexes = None

    def forward(self, predictions, masks, training):
        projection, *poses = predictions

        """
        Down/up samples the input to either the given size or the given scale_factor. The algorithm used for 
        interpolation is determined by mode. Currently temporal, spatial and volumetric sampling are supported, i.e. 
        expected inputs are 3-D, 4-D or 5-D in shape. The input dimensions are interpreted in the form: 
        mini-batch x channels x [optional depth] x [optional height] x width. The modes available for resizing are: 
        nearest, linear (3D-only), bi-linear, bicubic (4D-only), tri-linear (5D-only), area.
        """
        masks = F.interpolate(input=masks.unsqueeze(0), scale_factor=1/2, mode="bilinear", align_corners=True).squeeze()

        if not training:
            return dict(projection_loss=F.mse_loss(projection, masks, reduction="sum") / projection.size(0))

        ensemble_poses, student_poses = poses
        masks = repeat_tensor_for_each_element_in_batch(torch_tensor=masks, n=self.number_of_pose_predictor_candidates)

        projection_loss = F.mse_loss(projection, masks, reduction="none")
        projection_loss = projection_loss.sum((1, 2)).view(-1, self.num_candidates)

        minimum_indexes = projection_loss.argmin(dim=-1).detach()
        batch_indexes = torch.arange(minimum_indexes.size(0), device=minimum_indexes.device)

        # student loss
        minimum_projection_loss = projection_loss[batch_indexes, minimum_indexes].sum() / minimum_indexes.size(0)
        ensemble_poses = ensemble_poses.view(-1, self.number_of_pose_predictor_candidates, 4)

        best_poses = ensemble_poses[batch_indexes, minimum_indexes, :].detach()

        quaternion_operations = QuaternionOperations()

        poses_difference = F.normalize(
            quaternion_operations.quaternion_multiplication(q1=best_poses, q2=quaternion_operations.
                                                            quaternion_conjugate(q=student_poses)), dim=-1)

        angle_difference = poses_difference[:, 0]

        student_loss = (1 - angle_difference ** 2).sum() / minimum_indexes.size(0)

        # save to print histogram
        self.minimum_indexes = minimum_indexes.detach()

        total_loss = minimum_projection_loss + self.student_weight * student_loss

        return dict(projection_loss=minimum_projection_loss, student_loss=student_loss, total_loss=total_loss)
