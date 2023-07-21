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
from NoisyConv import NoisyConv, Conv

class UNHyperpriorEncoder(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, fpmodel, NoiseLayerNum, per_channel=True, **kwargs):
        N = fpmodel.N
        M = fpmodel.M
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a_noise = nn.Sequential(
            NoisyConv(fpmodel.g_a[0], per_channel = per_channel),
            fpmodel.g_a[1], # GDN保留
            NoisyConv(fpmodel.g_a[2], per_channel = per_channel),
            fpmodel.g_a[3],
            NoisyConv(fpmodel.g_a[4], per_channel = per_channel),
            fpmodel.g_a[5],
            NoisyConv(fpmodel.g_a[6], per_channel = per_channel),
        )

        self.g_a = nn.Sequential(
            Conv(fpmodel.g_a[0]),
            fpmodel.g_a[1], # GDN保留
            Conv(fpmodel.g_a[2]),
            fpmodel.g_a[3],
            Conv(fpmodel.g_a[4]),
            fpmodel.g_a[5],
            Conv(fpmodel.g_a[6]),
        )

        self.N = int(N)
        self.M = int(M)
        self.round = RoundSTE
        self.NoiseLayerNum = int(NoiseLayerNum)
        self.init_n()

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)
    
    def load_state_dict(self, state_dict):
        # update_registered_buffers(
        #     self.gaussian_conditional,
        #     "gaussian_conditional",
        #     ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
        #     state_dict,
        # )
        super().load_state_dict(state_dict)

    def init_n(self):
        for i, layer in enumerate(self.g_a_noise):
            if type(layer) == NoisyConv:
                if i <= self.NoiseLayerNum: layer.init_n(0)
                else: layer.init_n(1)

    def forward(self, x):
        y = self.round.apply(self.g_a(x))
        yy = self.round.apply(self.g_a_noise(x))

        return {
            "y": y,
            "yy": yy
        }
    
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.round()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output