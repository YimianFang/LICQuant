import torch
import torch.nn as nn

class NoisyConv(nn.Conv2d):
    def __init__(self, m:nn.Conv2d, per_channel = True):
        assert type(m) == nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=True if m.bias is not None else False,
            padding_mode=m.padding_mode)
        
        self.UNoise_n = nn.Parameter(torch.zeros(1))
        self.per_channel = per_channel
        self.weight = nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

    def init_n(self, state):
        x = self.weight.data
        if state == 0: # noise is added to the current layer or the layer behind
            if self.per_channel:
                self.UNoise_n = nn.Parameter(
                    x.detach().abs().mean(dim=list(range(1, x.dim()))) * 2 / (2 ** 4.5))
            else:
                self.UNoise_n = nn.Parameter(x.detach().abs().mean() * 2 / (2 ** 4.5))
        elif state == 1: # noise is added to the layer before
            if self.per_channel:
                self.UNoise_n = nn.Parameter(torch.zeros(x.detach().shape[0]))
            else:
                self.UNoise_n = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        UNoise = torch.empty_like(self.weight)
        n = self.UNoise_n
        if self.per_channel:
            for i in range(n.shape[0]):
                UNoise[i] = 2 * n[i] * torch.rand(self.weight[i].shape).to(n.device) - n[i]
        else:
           UNoise =  2 * n * torch.rand(self.weight.shape).to(n.device) - n
        return self._conv_forward(x, self.weight + UNoise, self.bias)

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