from compressai.models.google import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv, update_registered_buffers
from .utils import *

import torch
import torch.nn as nn
import torch.functional as F
import warnings
import copy
import os

class LSQPlusScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, fpmodel, **kwargs):
        N = fpmodel.N
        M = fpmodel.M
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            LSQPlusConv2d(fpmodel.g_a[0], signed=True),
            fpmodel.g_a[1], # GDN保留
            LSQPlusConv2d(fpmodel.g_a[2], signed=True),
            fpmodel.g_a[3],
            LSQPlusConv2d(fpmodel.g_a[4], signed=True),
            fpmodel.g_a[5],
            LSQPlusConv2d(fpmodel.g_a[6], signed=True),
        )

        self.g_s = nn.Sequential(
            LSQPlusConvTranspose2d(fpmodel.g_s[0], signed=True),
            fpmodel.g_s[1],
            LSQPlusConvTranspose2d(fpmodel.g_s[2], signed=True),
            fpmodel.g_s[3],
            LSQPlusConvTranspose2d(fpmodel.g_s[4], signed=True),
            fpmodel.g_s[5],
            LSQPlusConvTranspose2d(fpmodel.g_s[6], signed=True),
        )

        self.h_a = nn.Sequential(
            LSQPlusConv2d(fpmodel.h_a[0]),
            nn.ReLU(),
            LSQPlusConv2d(fpmodel.h_a[2]),
            nn.ReLU(),
            LSQPlusConv2d(fpmodel.h_a[4], signed=True),
        )

        self.h_s = nn.Sequential(
            LSQPlusConvTranspose2d(fpmodel.h_s[0]),
            nn.ReLU(),
            LSQPlusConvTranspose2d(fpmodel.h_s[2]),
            nn.ReLU(),
            LSQPlusConv2d(fpmodel.h_s[4]),
            nn.ReLU(),
        )

        self.entropy_bottleneck = fpmodel.entropy_bottleneck
        self.gaussian_conditional = fpmodel.gaussian_conditional
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)
    
    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }