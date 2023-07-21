import torch
import torch.nn as nn
import torch.nn.functional as F

class MixQConv(nn.Conv2d):
    def __init__(self, m:nn.Conv2d, default_precision=8, mixed_precision=16):
        assert type(m) == nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=True if m.bias is not None else False,
            padding_mode=m.padding_mode)
        
        self.default_precision = default_precision
        self.mixed_precision = mixed_precision
        self.weight = nn.Parameter(m.weight.detach())
        self.scale = torch.ones(self.weight.data.shape)
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

    def init_scale(self, mixed_channels=[]):
        x = self.weight.data
        Co, Ci, H, W = x.shape
        for i in range(Co):
            if i in mixed_channels: 
                precision = self.mixed_precision
            else: 
                precision = self.default_precision
            self.scale[i] = x[i].detach().abs().max().repeat(Ci*H*W).view(Ci, H, W) / 2 ** (precision - 1)

    def forward(self, x, Q=False):
        if Q:
            self.scale = self.scale.to(self.weight.device)
            return self._conv_forward(x, RoundSTE.apply(self.weight / self.scale) * self.scale, self.bias)
        else:
            return self._conv_forward(x, self.weight, self.bias)

class MixQTransConv(nn.ConvTranspose2d):
    def __init__(self, m:nn.ConvTranspose2d, default_precision=8, mixed_precision=16):
        assert type(m) == nn.ConvTranspose2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         output_padding=m.output_padding,
                         bias=True if m.bias is not None else False)
        
        self.default_precision = default_precision
        self.mixed_precision = mixed_precision
        self.weight = nn.Parameter(m.weight.detach())
        self.scale = torch.ones(self.weight.data.shape)
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

    def init_scale(self, mixed_channels=[]):
        x = self.weight.data.permute(1, 0, 2, 3)
        Co, Ci, H, W = x.shape
        for i in range(Co):
            if i in mixed_channels: 
                precision = self.mixed_precision
            else: 
                precision = self.default_precision
            self.scale[i] = x[i].detach().abs().max().repeat(Ci*H*W).view(Ci, H, W) / 2 ** (precision - 1)
        self.scale = self.scale.permute(1, 0, 2, 3)

    def forward(self, x, Q=False):
        if Q:
            self.scale = self.scale.to(self.weight.device)
            return F.conv_transpose2d(x, RoundSTE.apply(self.weight / self.scale) * self.scale, self.bias, self.stride,
                                self.padding, self.output_padding)
        else:
            return F.conv_transpose2d(x, self.weight, self.bias, self.stride,
                                self.padding, self.output_padding)
        
class Conv(nn.Conv2d):
    def __init__(self, m:nn.Conv2d):
        assert type(m) == nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=True if m.bias is not None else False,
            padding_mode=m.padding_mode)
        
        self.weight = nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

    def forward(self, x):
        return self._conv_forward(x, self.weight, self.bias)
    
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.round()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output