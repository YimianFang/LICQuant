import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.ops.parametrizers import NonNegativeParametrizer


class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out

class GDN_Taylor3(GDN):
    """
    GDN三阶泰勒展开
    """
    def __init__(self, N, inverse=False):
        super().__init__(N, inverse)
        self.state = True
        self.offset = nn.Parameter(torch.zeros_like(self.beta))
        self.de1 = nn.Parameter(torch.zeros_like(self.beta))
        self.de3 = nn.Parameter(torch.zeros_like(self.beta))
        self.thre = 1e-6

    def init_de1(self):
        beta = self.beta_reparam(self.beta)
        if self.inverse:
            self.de1 = nn.Parameter(torch.sqrt(beta))
        else:
            self.de1 = nn.Parameter(torch.rsqrt(beta))
            
    def forward(self, x: Tensor) -> Tensor:
        
        if self.state and self.de1.abs().sum() == 0:
            self.init_de1()
        self.state = False
        
        _, C, _, _ = x.size()
        
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        beta = self.beta_reparam(self.beta) #
        # beta = beta.reshape(1, -1, 1, 1) #
        offset = self.offset.reshape(1, -1, 1, 1)
        de1 = self.de1.reshape(1, -1, 1, 1)
        de3 = self.de3.reshape(1, -1, 1, 1)
        
        # if self.inverse:
        #     de1st = torch.sqrt(beta)
        #     de3rd = 1 / 2 * torch.rsqrt(beta) * F.conv2d(x**2, gamma)
        #     de5th = - 1 / 8 * torch.rsqrt(beta**3) * (F.conv2d(x**2, gamma)**2)
        # else:
        #     de1st = torch.rsqrt(beta)
        #     de3rd = - 1 / 2 * torch.rsqrt(beta**3) * F.conv2d(x**2, gamma)
        #     de5th = 3 / 8 * torch.rsqrt(beta**5) * (F.conv2d(x**2, gamma)**2)
        
        # if self.inverse:
        #     de3 = torch.clamp_min(de3, torch.tensor(self.thre).to(de3.device))
        # else:
        #     de3 = torch.clamp_max(de3, torch.tensor(-self.thre).to(de3.device))
            
        if self.inverse:
            out = de1 * x * F.conv2d(x.abs(), gamma, beta) + de3 * F.conv2d(x**2, gamma, beta) + offset
        else:
            out = de1 * x + de3 * F.conv2d(x**2, gamma, beta) + offset
            
        # out = x * (de1st + de3rd + de5th) #

        return out

class GDN_v2(GDN):

    def __init__(self, N, inverse=False):
        super().__init__(N, inverse)
        self.state = True
        self.offset = nn.Parameter(torch.zeros_like(self.beta))
        self.s1 = nn.Parameter(torch.ones_like(self.beta))
        self.s3 = nn.Parameter(torch.zeros_like(self.beta))
        self.momentum = 0.9
            
    def forward(self, x: Tensor) -> Tensor:
        
        _, C, _, _ = x.size()
        
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        beta = self.beta_reparam(self.beta) #
        # beta = beta.reshape(1, -1, 1, 1) #
        s1 = self.s1.reshape(1, -1, 1, 1)
        
        if self.state and not self.training:
            self.s3.data = torch.sqrt(F.conv2d(x**2, gamma, beta)).amax(dim=(0, 2, 3))
            s3 = self.s3.reshape(1, -1, 1, 1)
            if self.inverse:
                self.offset.data = (x * torch.sqrt(F.conv2d(x**2, gamma, beta)) - x * s3).mean(dim=(0, 2, 3))
            else:
                self.offset.data = (x / torch.sqrt(F.conv2d(x**2, gamma, beta)) - x / s3).mean(dim=(0, 2, 3))
            self.state = False
        # elif self.training:
        #     self.s3.data = self.s3.data * self.momentum + torch.sqrt(F.conv2d(x**2, gamma, beta)).amax(dim=(0, 2, 3))
        s3 = self.s3.reshape(1, -1, 1, 1)
        offset = self.offset.reshape(1, -1, 1, 1)
        
        if self.inverse:
            out = s1 * x * s3 + offset
        else:
            out = s1 * x / s3 + offset
            
        # out = x * (de1st + de3rd + de5th) #

        return out
    
class GDN1(GDN):
    r"""Simplified GDN layer.

    Introduced in `"Computationally Efficient Neural Image Compression"
    <http://arxiv.org/abs/1912.08771>`_, by Johnston Nick, Elad Eban, Ariel
    Gordon, and Johannes Ballé, (2019).

    .. math::

        y[i] = \frac{x[i]}{\beta[i] + \sum_j(\gamma[j, i] * |x[j]|}

    """

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(torch.abs(x), gamma, beta)

        if not self.inverse:
            norm = 1.0 / norm

        out = x * norm

        return out

class GDN_exp(GDN):

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        out = x * torch.exp(-torch.abs(norm))

        return out