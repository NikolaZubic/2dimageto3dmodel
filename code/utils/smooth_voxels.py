"""

Author: Nikola Zubic
"""
import torch
import numpy as np
import torch.nn.functional as F


class VoxelsSmooth(object):
    def __init__(self):
        print("Voxels Smooth called.")

    def separate_kernels(self, std_dev, kernel_size=21):
        """
        Gaussian kernel is separable, so we can express 3D convolution function as the combination of three 1D
        convolutions. For large filters, Fourier transform of image and filter is applicable, multiply these results
        and then take the inverse Fourier transform.

        :param std_dev: standard deviation
        :param kernel_size: kernel size
        :return: three separate Gaussian kernels
        """
        a, b = (-kernel_size // 2, kernel_size // 2)

        x = torch.arange(a + 1.0, b + 1.0)

        # Gaussian filter modifies the input signal by convolution with a Gaussian function
        kernel_1d = torch.exp(pow(-x, 2) / (2 * pow(std_dev, 2)))
        # Normalize components
        kernel_1d = kernel_1d / kernel_1d.sum()

        """
        PyTorch allows a tensor to be a View of an existing tensor. View tensor shares the same underlying data with its
        base tensor. Supporting View avoids explicit data copy, thus allows us to do fast and memory efficient 
        reshaping, slicing and element-wise operations.
        """
        first_kernel = kernel_1d.view(1, 1, 1, 1, -1)  # length
        second_kernel = kernel_1d.view(1, 1, 1, -1, 1)  # column
        third_kernel = kernel_1d.view(1, 1, -1, 1, 1)  # depth

        return [first_kernel, second_kernel, third_kernel]

    def smooth(self, voxels, kernels, scale=None):
        """
        Applies Gaussian blur ( https://en.wikipedia.org/wiki/Gaussian_blur ) to voxels with separated kernels and
        scales them. Here, we use Gaussian smoothing of a voxel in order to enhance voxel structures at different
        scales.

        :param voxels: voxels for smoothing
        :param kernels: separated Gaussian kernels
        :param scale: scale factor
        :return: smoothed voxels
        """
        convolved_voxels = None

        # add channel for convolutions (RGB)
        colours = voxels.size(0)
        # https://www.kaggle.com/shivamb/3d-convolutions-understanding-use-case
        voxels = voxels.unsqueeze(0)

        """
        Reordering batch and channels by using groups was necessary because standard convolutional 3d operator had
        problems.
        """
        for kernel in kernels:
            # add padding for kernel dimension
            padding = 3 * [0]
            padding[np.argmax(kernel.shape) - 2] = max(kernel.shape) // 2

            # Applies a 3D convolution over an input signal composed of several input planes.
            convolved_voxels = F.conv3d(input=voxels, weight=kernel.repeat(colours, 1, 1, 1, 1), stride=1,
                                        padding=padding, groups=colours)
            """
            split input into groups, in_channels should be divisible by the number of groups
            """

        convolved_voxels = convolved_voxels.squeeze(0)

        if scale is not None:
            convolved_voxels = convolved_voxels * scale.view(-1, 1, 1, 1)
            convolved_voxels = convolved_voxels.clamp(0, 1)

        return convolved_voxels
