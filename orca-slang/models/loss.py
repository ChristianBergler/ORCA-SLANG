"""
Module: loss.py
Authors: Christian Bergler, Manuel Schmitt, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""


import torch
import torch.nn as nn


class DeepFeatureLoss(nn.Module):
    """Reduces the loss for parts where the ground truth spectrogram is 0 for all
       frequencies.
    """

    def __init__(self, freq_dim=-1, reduction="elementwise_mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.freq_dim = freq_dim
        self.reduction = reduction

    def __call__(self, recon_x, x):
        mse_ = self.mse(recon_x, x).sum(dim=self.freq_dim)
        mse_tmp = mse_.clone()
        relevant = torch.where(
            torch.gt(x.sum(dim=self.freq_dim), 0.), mse_, mse_tmp.mul_(0.1)
        )
        if self.reduction == "elementwise_mean":
            relevant = relevant.mean()
        elif self.reduction == "sum":
            relevant = relevant.sum()

        return relevant
