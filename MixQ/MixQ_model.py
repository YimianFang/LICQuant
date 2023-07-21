from compressai.models.google import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv, update_registered_buffers

import torch
import torch.nn as nn
import torch.functional as F
import warnings
import copy
import os
from MixQConv import MixQConv, Conv, MixQTransConv, RoundSTE

class MixQHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, fpmodel, default_precision=8, mixed_precision=16, **kwargs):
        N = fpmodel.N
        M = fpmodel.M
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.default_precision = default_precision
        self.mixed_precision = mixed_precision
        self.QLayerList = []
        self.g_a = nn.Sequential(
            MixQConv(fpmodel.g_a[0], default_precision, mixed_precision),
            fpmodel.g_a[1], # GDN保留
            MixQConv(fpmodel.g_a[2], default_precision, mixed_precision),
            fpmodel.g_a[3],
            MixQConv(fpmodel.g_a[4], default_precision, mixed_precision),
            fpmodel.g_a[5],
            MixQConv(fpmodel.g_a[6], default_precision, mixed_precision),
        )

        self.g_s = nn.Sequential(
            MixQTransConv(fpmodel.g_s[0], default_precision, mixed_precision),
            fpmodel.g_s[1],
            MixQTransConv(fpmodel.g_s[2], default_precision, mixed_precision),
            fpmodel.g_s[3],
            MixQTransConv(fpmodel.g_s[4], default_precision, mixed_precision),
            fpmodel.g_s[5],
            MixQTransConv(fpmodel.g_s[6], default_precision, mixed_precision),
        )

        self.h_a = nn.Sequential(
            MixQConv(fpmodel.h_a[0], default_precision, mixed_precision),
            nn.ReLU(),
            MixQConv(fpmodel.h_a[2], default_precision, mixed_precision),
            nn.ReLU(),
            MixQConv(fpmodel.h_a[4], default_precision, mixed_precision),
        )

        self.h_s = nn.Sequential(
            MixQTransConv(fpmodel.h_s[0], default_precision, mixed_precision),
            nn.ReLU(),
            MixQTransConv(fpmodel.h_s[2], default_precision, mixed_precision),
            nn.ReLU(),
            MixQConv(fpmodel.h_s[4], default_precision, mixed_precision),
            nn.ReLU(),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.round = RoundSTE

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

    def g_a_layer_init_scale(self, QLayerList=[], mixed_channels={}):
        assert type(mixed_channels) == dict and type(QLayerList) == list
        self.QLayerList = QLayerList
        for l in QLayerList:
            assert  type(self.g_a[l]) == MixQConv
            self.g_a[l].init_scale(mixed_channels[l])

    def forward(self, x, Q=False):
        y = self.round.apply(self.g_a(x))
        yy = x
        for i in range(len(self.g_a)):
            if i in self.QLayerList and type(self.g_a[i]) == MixQConv:
                yy = self.g_a[i](yy, Q)
            else:
                yy = self.g_a[i](yy)

        z = self.h_a(torch.abs(yy))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(yy, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "y": y,
            "yy": self.round.apply(yy),
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
