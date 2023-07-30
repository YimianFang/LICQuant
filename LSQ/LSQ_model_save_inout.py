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
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            QuanConv2d(fpmodel.g_a[0], LsqQuan(8), LsqQuan(8, per_channel=False)),
            fpmodel.g_a[1], # GDN保留
            QuanConv2d(fpmodel.g_a[2], LsqQuan(8), LsqQuan(8, per_channel=False)),
            fpmodel.g_a[3],
            QuanConv2d(fpmodel.g_a[4], LsqQuan(8), LsqQuan(8, per_channel=False)),
            fpmodel.g_a[5],
            QuanConv2d(fpmodel.g_a[6], LsqQuan(8), LsqQuan(8, per_channel=False)),
        )

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
        if not os.path.exists("savedata/LSQ_GDN_inout"):
            os.mkdir("savedata/LSQ_GDN_inout")
        for l, m in enumerate(self.g_a):
            if type(m) == GDN:
                torch.save(x, "savedata/LSQ_GDN_inout/g_a." + str(l) + ".in.pth.tar")
                x = m(x)
                torch.save(x, "savedata/LSQ_GDN_inout/g_a." + str(l) + ".out.pth.tar")
            else:
                x = m(x)
        y = x
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        for l, m in enumerate(self.g_s):
            if type(m) == GDN:
                torch.save(y_hat, "savedata/LSQ_GDN_inout/g_s." + str(l) + ".in.pth.tar")
                y_hat = m(y_hat)
                torch.save(y_hat, "savedata/LSQ_GDN_inout/g_s." + str(l) + ".out.pth.tar")
            else:
                y_hat = m(y_hat)
        x_hat = y_hat

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }