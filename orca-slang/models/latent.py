"""
Module: latent.py
Authors: Christian Bergler, Manuel Schmitt, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""


from torch import nn

from models.utils import get_padding

DefaultLatentOpts = {
    "in_channels": 1,
    "latent_channels": 1,
    "adaptive_pooling": (8, 16),
    "latent_kernel_size": 1,
    "latent_kernel_stride": 1,
    "latent_size": 512
}

class Latent(nn.Module):
    def __init__(self, opts: dict = DefaultLatentOpts):
        super().__init__()
        self._opts = opts
        self.conv1 = nn.Conv2d(
            opts["in_channels"],
            opts["latent_channels"],
            kernel_size=opts["latent_kernel_size"],
            stride=opts["latent_kernel_stride"],
            padding=get_padding(opts["latent_kernel_size"]),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(opts["latent_channels"])
        self.relu1 = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d((8, 16), return_indices=True)

        #dense layer according to latent size
        if opts.get("latent_size") is not None and opts["latent_size"] != 512:
            self.dense_cmpr = nn.Linear(opts["in_channels"], opts["latent_size"])
            self.dense_decmpr = nn.Linear(opts["latent_size"], opts["in_channels"])
        else:
            self.dense_cmpr = None
            self.dense_decmpr = None

        self.max_unpool = nn.MaxUnpool2d((8, 16))

        self.conv2 = nn.ConvTranspose2d(
            opts["latent_channels"],
            opts["in_channels"],
            kernel_size=opts["latent_kernel_size"],
            stride=opts["latent_kernel_stride"],
            padding=get_padding(opts["latent_kernel_size"]),
            output_padding=get_padding(opts["latent_kernel_stride"]),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(opts["in_channels"])
        self.relu2 = nn.ReLU(inplace=True)

        self._layer_output = dict()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        z = self.relu1(x)

        z, indices = self.max_pool(z)

        hidden_layer = z.clone()

        if self.dense_cmpr is None and self.dense_decmpr is None:
            self._layer_output["code"] = hidden_layer.view(hidden_layer.size(0), -1)
        else:
            hidden_layer = self.dense_cmpr(hidden_layer.view(hidden_layer.size(0), -1))
            self._layer_output["code"] = hidden_layer.view(hidden_layer.size(0), -1)
            z = self.dense_decmpr(hidden_layer).unsqueeze(-1).unsqueeze(-1)

        x = self.max_unpool(z, indices)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x, z

    def get_layer_output(self):
        return self._layer_output

    def model_opts(self):
        return self._opts

