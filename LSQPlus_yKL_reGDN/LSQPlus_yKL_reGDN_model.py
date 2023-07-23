from compressai.models.google import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import conv, deconv, update_registered_buffers
from reGDN import *
from .utils import *

import torch
import torch.nn as nn
import torch.functional as F
import warnings
import copy
import os

class LSQPlus_reGDN_ScaleHyperprior(CompressionModel):
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
            LSQPlusConv2d(fpmodel.g_a[0], signed=True),
            GDN_v6(N), # GDN保留
            LSQPlusConv2d(fpmodel.g_a[2], signed=True),
            GDN_v6(N),
            LSQPlusConv2d(fpmodel.g_a[4], signed=True),
            GDN_v6(N),
            LSQPlusConv2d(fpmodel.g_a[6], signed=True),
        )

        # self.g_a_fp = copy.deepcopy(fpmodel.g_a)
        
        self.g_a_fp = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )
        
        self.g_a_fp[0].weight = fpmodel.g_a[0].weight
        self.g_a_fp[2].weight = fpmodel.g_a[2].weight
        self.g_a_fp[4].weight = fpmodel.g_a[4].weight
        self.g_a_fp[6].weight = fpmodel.g_a[6].weight

        self.g_s = nn.Sequential(
            LSQPlusConvTranspose2d(fpmodel.g_s[0], signed=True),
            GDN_v6(N, inverse=True),
            LSQPlusConvTranspose2d(fpmodel.g_s[2], signed=True),
            GDN_v6(N, inverse=True),
            LSQPlusConvTranspose2d(fpmodel.g_s[4], signed=True),
            GDN_v6(N, inverse=True),
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