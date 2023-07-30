import torch as t

from .quantizer import Quantizer

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class ActStaQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, method="avg"):
        super().__init__(bit)

        self.all_positive = all_positive
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.val = t.nn.Parameter(t.zeros(1), requires_grad=False)
        self.cnt = t.nn.Parameter(t.zeros(1), requires_grad=False)
        self.bit = bit
        self.method = method

    def sta_getscale(self, x):
        if self.training:
            if self.all_positive:
                cur_val = x.detach().max()
            else:
                cur_val = x.detach().abs().max()
            assert self.method == "avg" or self.method == "max"
            if self.method == "avg":
                self.val.data = (self.val.data * self.cnt.data + cur_val) / (self.cnt.data + 1)
            elif self.method == "max":
                self.val.data = max(self.val.data, cur_val)
            self.cnt.data += 1
        if self.all_positive:
            return self.val.data / (2 ** self.bit - 1)
        else:
            return self.val.data / 2 ** (self.bit - 1)

    def forward(self, x):
        s_scale = self.sta_getscale(x)
        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x
    
class WQuan(Quantizer):
    def __init__(self, bit, symmetric=False, per_channel=True):
        super().__init__(bit)

        if symmetric:
            # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            self.thd_neg = - 2 ** (bit - 1) + 1
            self.thd_pos = 2 ** (bit - 1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.thd_neg = - 2 ** (bit - 1)
            self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.bit = bit

    def get_scale(self, x):
        if self.per_channel:
           xx = x.detach().abs()
           xx = xx.reshape(xx.shape[0], -1).max(dim=1)[0] / 2 ** (self.bit - 1)
           return xx.reshape(-1, 1, 1, 1)
        else:
            return x.detach().abs().max() / 2 ** (self.bit - 1)

    def forward(self, x):
        s_scale = self.get_scale(x)
        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x