from compressai.models.google import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv, update_registered_buffers
from reGDN import GDN_Taylor3

import torch
import torch.nn as nn
import torch.functional as F
import warnings
import copy
import os

class reGDN_ScaleHyperprior(CompressionModel):
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
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            fpmodel.g_a[0],
            GDN_Taylor3(N),
            fpmodel.g_a[2],
            GDN_Taylor3(N),
            fpmodel.g_a[4],
            GDN_Taylor3(N),
            fpmodel.g_a[6],
        )

        self.g_s = nn.Sequential(
            fpmodel.g_s[0],
            GDN_Taylor3(N, inverse=True),
            fpmodel.g_s[2],
            GDN_Taylor3(N, inverse=True),
            fpmodel.g_s[4],
            GDN_Taylor3(N, inverse=True),
            fpmodel.g_s[6],
        )

        self.h_a = nn.Sequential(
            fpmodel.h_a[0],
            nn.ReLU(inplace=True),
            fpmodel.h_a[2],
            nn.ReLU(inplace=True),
            fpmodel.h_a[4],
        )

        self.h_s = nn.Sequential(
            fpmodel.h_s[0],
            nn.ReLU(inplace=True),
            fpmodel.h_s[2],
            nn.ReLU(inplace=True),
            fpmodel.h_s[4],
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
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
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
class roundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = torch.round(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        # print("roundSTE_grad: ", grad_output)
        return grad_output, None, None, None