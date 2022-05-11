"""
Module: residual_decoder.py
Authors: Christian Bergler, Manuel Schmitt, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""


import torch.nn as nn
from models.residual_base import ResidualBase, get_block_sizes, get_block_type
from models.utils import get_padding

DefaultDecoderOpts = {
    "output_channels": 1,
    "conv_kernel_size": 7,
    "input_channels": 512,
    "resnet_size": 18,
    "output_activation": "sigmoid",
    "output_stride": (2, 2),
    "dropout_prob_decoder": 0.2,
}

"""
Defines the convolutional or feature extraction part (residual layers) of the CNN. According to the chosen and supported 
ResNet architecture (ResNet18, 34, 50, 101, 152) the block types and sizes are generated in order to construct the 
respective residual encoder part as well as the backweard path computation.
"""
class ResidualDecoder(ResidualBase):
    def __init__(self, opts: dict = DefaultDecoderOpts):
        super().__init__()
        self._opts = opts
        self._layer_output = dict()
        self.cur_in_ch = opts["input_channels"]
        self.block_sizes = get_block_sizes(opts["resnet_size"])
        self.block_type = get_block_type(opts["resnet_size"])

        # Upsample ResNet layers use transposed convolution when stride > 1
        self.layer1 = self.make_layer(
            self.block_type, 256, self.block_sizes[3], (2, 2), "upsample"
        )  # 512 -> 256
        if opts["dropout_prob_decoder"] > 0.0:
            self.l1_dropout = nn.Dropout(p=opts["dropout_prob_decoder"])

        self.layer2 = self.make_layer(
            self.block_type, 128, self.block_sizes[2], (2, 2), "upsample"
        )  # 256 -> 128
        if opts["dropout_prob_decoder"] > 0.0:
            self.l2_dropout = nn.Dropout(p=opts["dropout_prob_decoder"])

        self.layer3 = self.make_layer(
            self.block_type, 64, self.block_sizes[1], (2, 2), "upsample"
        )  # 128 -> 64
        if opts["dropout_prob_decoder"] > 0.0:
            self.l3_dropout = nn.Dropout(p=opts["dropout_prob_decoder"])

        self.layer4 = self.make_layer(
            self.block_type, 64, self.block_sizes[0], (2, 2), "upsample"
        )  # 64 -> 64; use stride 2 instead of max unpooling
        if opts["dropout_prob_decoder"] > 0.0:
            self.l4_dropout = nn.Dropout(p=opts["dropout_prob_decoder"])

        self.conv_out = nn.ConvTranspose2d(
            in_channels=64 * self.block_type.expansion,
            out_channels=opts["output_channels"],
            kernel_size=opts["conv_kernel_size"],
            padding=get_padding(opts["conv_kernel_size"]),
            output_padding=get_padding(opts["output_stride"]),
            stride=opts["output_stride"],
            bias=False,
        )

        if opts["output_activation"].lower() == "sigmoid":
            self.activation_out = nn.Sigmoid()
        elif opts["output_activation"].lower() == "relu":
            self.activation_out = nn.ReLU(inplace=True)
        elif opts["output_activation"].lower() == "tanh":
            self.activation_out = nn.Tanh()
        elif opts["output_activation"].lower() == "none":
            self.activation_out = lambda x: x
        else:
            raise NotImplementedError(
                "Unsupported output activation: {}".format(opts["output_activation"])
            )

    def forward(self, x):
        x, z = x

        self._layer_output["input_layer"] = x
        x = self.layer1(x)
        if self._opts["dropout_prob_decoder"] > 0.0:
            x = self.l1_dropout(x)

        self._layer_output["residual_layer1"] = x
        x = self.layer2(x)
        if self._opts["dropout_prob_decoder"] > 0.0:
            x = self.l2_dropout(x)

        self._layer_output["residual_layer2"] = x
        x = self.layer3(x)
        if self._opts["dropout_prob_decoder"] > 0.0:
            x = self.l3_dropout(x)

        self._layer_output["residual_layer3"] = x
        x = self.layer4(x)
        if self._opts["dropout_prob_decoder"] > 0.0:
            x = self.l4_dropout(x)

        self._layer_output["residual_layer4"] = x
        x = self.conv_out(x)
        x = self.activation_out(x)

        self._layer_output["output_layer"] = x

        if z is not None:
            x = (x, z)

        return x

    def model_opts(self):
        return self._opts

    def get_layer_output(self):
        return self._layer_output
