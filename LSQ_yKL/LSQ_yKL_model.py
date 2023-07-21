from compressai.models.google import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv, update_registered_buffers
from utils import *

import torch
import torch.nn as nn
import torch.functional as F
import warnings
import copy
import os

class LSQScaleHyperprior(CompressionModel):
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
            QuanConv2d(fpmodel.g_a[0], LsqQuan(8), LsqQuan(8, per_channel=False)),
            fpmodel.g_a[1], # GDN保留
            QuanConv2d(fpmodel.g_a[2], LsqQuan(8), LsqQuan(8, per_channel=False)),
            fpmodel.g_a[3],
            QuanConv2d(fpmodel.g_a[4], LsqQuan(8), LsqQuan(8, per_channel=False)),
            fpmodel.g_a[5],
            QuanConv2d(fpmodel.g_a[6], LsqQuan(8), LsqQuan(8, per_channel=False)),
        )

        self.g_a_fp = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )
        
        self.g_a_fp[0].weight = self.g_a[0].weight
        self.g_a_fp[2].weight = self.g_a[2].weight
        self.g_a_fp[4].weight = self.g_a[4].weight
        self.g_a_fp[6].weight = self.g_a[6].weight

        self.g_s = nn.Sequential(
            QuanConvTranspose2d(fpmodel.g_s[0], LsqQuan(8), LsqQuan(8, per_channel=False)),
            fpmodel.g_s[1],
            QuanConvTranspose2d(fpmodel.g_s[2], LsqQuan(8), LsqQuan(8, per_channel=False)),
            fpmodel.g_s[3],
            QuanConvTranspose2d(fpmodel.g_s[4], LsqQuan(8), LsqQuan(8, per_channel=False)),
            fpmodel.g_s[5],
            QuanConvTranspose2d(fpmodel.g_s[6], LsqQuan(8), LsqQuan(8, per_channel=False)),
        )

        self.h_a = nn.Sequential(
            QuanConv2d(fpmodel.h_a[0], LsqQuan(8), LsqQuan(8, all_positive=True, per_channel=False)),
            nn.ReLU(),
            QuanConv2d(fpmodel.h_a[2], LsqQuan(8), LsqQuan(8, all_positive=True, per_channel=False)),
            nn.ReLU(),
            QuanConv2d(fpmodel.h_a[4], LsqQuan(8), LsqQuan(8, per_channel=False)),
        )

        self.h_s = nn.Sequential(
            QuanConvTranspose2d(fpmodel.h_s[0], LsqQuan(8), LsqQuan(8, all_positive=True, per_channel=False)),
            nn.ReLU(),
            QuanConvTranspose2d(fpmodel.h_s[2], LsqQuan(8), LsqQuan(8, all_positive=True, per_channel=False)),
            nn.ReLU(),
            QuanConv2d(fpmodel.h_s[4], LsqQuan(8), LsqQuan(8, all_positive=True, per_channel=False)),
            nn.ReLU(),
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
        y_fp = self.g_a_fp(x)
        y_fp_hat = roundSTE.apply(y_fp)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        y_hat = roundSTE.apply(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "y_fp_hat": y_fp_hat,
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