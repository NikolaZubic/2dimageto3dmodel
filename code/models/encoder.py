# author: Nikola Zubic

import torch.nn as nn
from basic_blocks import BasicBlocks


class Encoder(nn.Module):
    """
    Almost the same architecture like in original paper, but with added ReLU activation with Kaiming Initialization and
    batch-norm.
    """
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        feature_size = self.image_size // 8

        self.network_architecture = nn.Sequential(
            BasicBlocks.convolutional(input_channels=3, output_channels=16, kernel_size=5, stride=2, padding=2,
                                      bias=True, use_activation_function=True),
            BasicBlocks.convolutional(input_channels=16, output_channels=16, kernel_size=3, stride=2, padding=1,
                                      bias=True, use_activation_function=True),
            BasicBlocks.convolutional(input_channels=16, output_channels=16, kernel_size=3, stride=1, padding=1,
                                      bias=True, use_activation_function=True),
            BasicBlocks.convolutional(input_channels=16, output_channels=16, kernel_size=3, stride=2, padding=1,
                                      bias=True, use_activation_function=True),
            BasicBlocks.convolutional(input_channels=16, output_channels=16, kernel_size=3, stride=1, padding=1,
                                      bias=True, use_activation_function=True),
            BasicBlocks.convolutional(input_channels=16, output_channels=16, kernel_size=3, stride=2, padding=1,
                                      bias=True, use_activation_function=True),
            BasicBlocks.convolutional(input_channels=16, output_channels=16, kernel_size=3, stride=1, padding=1,
                                      bias=True, use_activation_function=True),
            BasicBlocks.convolutional(input_channels=16, output_channels=16, kernel_size=3, stride=2, padding=1,
                                      bias=True, use_activation_function=True),
            BasicBlocks.convolutional(input_channels=16, output_channels=16, kernel_size=3, stride=1, padding=1,
                                      bias=True, use_activation_function=True)
        )

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size ** 2, 1024, bias=True),
            nn.ReLU(True),
            nn.Linear(1024, 1024)
        )

    def forward(self, image):
        convolutional_features = self.network_architecture(image)
        features = self.features(convolutional_features)

        return features
