# author: Nikola Zubic

import torch
import torch.nn as nn


class BasicBlocks(object):

    @staticmethod
    def convolutional(input_channels, output_channels, kernel_size, stride, padding, bias,
                      use_activation_function):
        convolution_2d = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=bias)

        func = None

        if use_activation_function:
            func = nn.ReLU(True)
        else:
            func = nn.Identity()

        # perform batch normalization to results if there's no bias
        normalized = None
        if not bias:
            normalized = nn.BatchNorm2d(output_channels)
        else:
            normalized = nn.Identity()  # nothing has changed otherwise

        return nn.Sequential(
            convolution_2d,
            func,
            normalized
        )

    @staticmethod
    def pose_prediction(m):
        """
        Prediction of pose with 2 hidden layers.

        :param m: number of features
        :return: prediction network
        """
        return nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(True),
            nn.Linear(m, m),
            nn.ReLU(True),
            nn.Linear(m, 4)
        )
