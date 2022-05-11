"""
Module: residual_encoder.py
Authors: Christian Bergler, Manuel Schmitt, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import torch.nn as nn

from models.residual_base import ResidualBase, get_block_sizes, get_block_type
from models.utils import get_padding


# ResNet 18 default
DefaultEncoderOpts = {
    "input_channels": 1,
    "conv_kernel_size": 7,
    "max_pool": 2,
    "resnet_size": 18,
    "dropout_prob_encoder": 0.2,
}

"""
Defines the convolutional or feature extraction part (residual layers) of the CNN. According to the chosen and supported 
ResNet architecture (ResNet18, 34, 50, 101, 152) the block types and sizes are generated in order to construct the 
respective residual encoder part as well as the forward path computation.
"""

class ResidualEncoder(ResidualBase):
    def __init__(self, opts: dict = DefaultEncoderOpts):
        super().__init__()
        self._opts = opts
        self._layer_output = dict()
        self.cur_in_ch = 64
        self.block_sizes = get_block_sizes(opts["resnet_size"])
        self.block_type = get_block_type(opts["resnet_size"])

        self.conv1 = nn.Conv2d(
            opts["input_channels"],
            out_channels=64,
            kernel_size=opts["conv_kernel_size"],
            padding=get_padding(opts["conv_kernel_size"]),
            stride=(2, 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.cur_in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        # Not the default
        if opts["max_pool"] == 1:
            self.max_pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=get_padding(3)
            )
            stride1 = (1, 1)  # Stride of first ResNet layer
        # Not the default
        elif opts["max_pool"] == 0:
            self.max_pool = None
            stride1 = (2, 2)  # Reduce dim in first ResNet layer instead
        # THE DEFAULT:
        elif opts["max_pool"] == 2:
            self.max_pool = None
            stride1 = (1, 1)

        self.layer1 = self.make_layer(self.block_type, 64, self.block_sizes[0], stride1)
        if opts["dropout_prob_encoder"] > 0.0:
            self.l1_dropout = nn.Dropout(p=opts["dropout_prob_encoder"])

        self.layer2 = self.make_layer(self.block_type, 128, self.block_sizes[1], (2, 2))
        if opts["dropout_prob_encoder"] > 0.0:
            self.l2_dropout = nn.Dropout(p=opts["dropout_prob_encoder"])

        self.layer3 = self.make_layer(self.block_type, 256, self.block_sizes[2], (2, 2))
        if opts["dropout_prob_encoder"] > 0.0:
            self.l3_dropout = nn.Dropout(p=opts["dropout_prob_encoder"])

        self.layer4 = self.make_layer(self.block_type, 512, self.block_sizes[3], (2, 2))
        if opts["dropout_prob_encoder"] > 0.0:
            self.l4_dropout = nn.Dropout(p=opts["dropout_prob_encoder"])

    def forward(self, x):
        self._layer_output["input_layer"] = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        self._layer_output["init_conv_layer"] = x
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.layer1(x)
        if self._opts["dropout_prob_encoder"] > 0.0:
            x = self.l1_dropout(x)

        self._layer_output["residual_layer1"] = x
        x = self.layer2(x)
        if self._opts["dropout_prob_encoder"] > 0.0:
            x = self.l2_dropout(x)

        self._layer_output["residual_layer2"] = x
        x = self.layer3(x)
        if self._opts["dropout_prob_encoder"] > 0.0:
            x = self.l3_dropout(x)

        self._layer_output["residual_layer3"] = x
        x = self.layer4(x)
        if self._opts["dropout_prob_encoder"] > 0.0:
            x = self.l4_dropout(x)

        self._layer_output["residual_layer4"] = x

        return x

    def model_opts(self):
        return self._opts

    def get_layer_output(self):
        return self._layer_output
