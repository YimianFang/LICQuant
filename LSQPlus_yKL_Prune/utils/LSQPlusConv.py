import math
from enum import Enum

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

class LSQPlus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, beta, lower_bound, upper_bound, wora, scale_per_channel, beta_per_channel):
        r"""
        LSQPlus: y = Round[clamp((x - beta)/s, n, p)] * s + beta
        Args:
            x: input tensor
            scale: QAT scale
            beta: QAT offset
        return:
            x_quant: quantization tensor of x
        """
        x_hat = ((x - beta) / scale).round()
        ctx.save_for_backward(x, x_hat, scale, beta)
        ctx.constant = [lower_bound, upper_bound, wora, scale_per_channel, beta_per_channel]
        x_hat = x_hat.clamp(lower_bound, upper_bound)
        x_quant = x_hat * scale + beta
        return x_quant
    
    @staticmethod
    def backward(ctx, grad_output):
        r"""

        Backward:
            x_gradient: x[x > p or x < n] = 0 else 1
            scale_gradient: -(x - beta)/s + round[(x - beta)/s] else n or p
            beta_gradient: 0 or 1
        """
        x, x_hat, scale, beta = ctx.saved_variables
        lower_bound, upper_bound, wora, scale_per_channel, beta_per_channel = ctx.constant

        r"""1. input gradient"""
        x_grad = torch.ones_like(grad_output)
        x_grad[(x - beta)/scale <= lower_bound] = 0
        x_grad[(x - beta)/scale >= upper_bound] = 0
        x_grad *= grad_output

        r"""2. scale gradient"""
        scale_grad = -(x - beta)/scale + x_hat
        scale_grad[(x - beta)/scale <= lower_bound] = float(lower_bound)
        scale_grad[(x - beta)/scale >= upper_bound] = float(upper_bound)
        if scale_per_channel:
            scale_grad = (scale_grad * grad_output).sum(dim=list(range(1, x.dim())), keepdim=True) / math.sqrt(x.numel() * upper_bound) 
        else:
            scale_grad = (scale_grad * grad_output).sum().unsqueeze(dim=0) / math.sqrt(x.numel() * upper_bound) 

        r"""3. offset gradient"""
        beta_grad = torch.zeros_like(x)
        beta_grad[(x-beta)/scale <= lower_bound] = 1
        beta_grad[(x-beta)/scale >= upper_bound] = 1
        if beta_per_channel:
            if wora == "w":
                beta_grad = (beta_grad * grad_output).sum(dim=list(range(1, x.dim())), keepdim=True) / math.sqrt(x.numel() * upper_bound) 
            else:
                beta_grad = (beta_grad * grad_output).sum(dim=[0] + list(range(2, x.dim())), keepdim=True) / math.sqrt(x.numel() * upper_bound) 
        else:
            beta_grad = (beta_grad * grad_output).sum().unsqueeze(dim=0) / math.sqrt(x.numel() * upper_bound) 

        return x_grad, scale_grad, beta_grad, None, None, None, None, None

class LSQPlusConv2d(nn.Conv2d):
    """Generates quantized convolutional layers.

args:
    nbits_w(int): bitwidth for the weight quantization,
    nbits_a(int): bitwidth for the activation quantization,
procedure:
    g = 1.0 / math.sqrt(x.numel() * Qp) if self.scale_grad else 1.0

    act_alpha = grad_scale(self.act_alpha, g) if self.training else self.running_act_alpha

    if not self.add_offset:

        x_q = round_pass((x / act_alpha ).clamp(Qn, Qp)) * act_alpha 

    else:

        act_offset = grad_scale(self.act_offset, g) if self.training else self.running_act_offset
        
        x_q = round_pass((x - act_offset) / act_alpha).clamp(Qn, Qp) * act_alpha + act_offset
    """
    def __init__(
        self,
        fpmodule,
        nbits_w = 8,
        nbits_a = 8,
        signed = False,
        quant_activation = True,
        add_offset = True,
        per_channel = True,
        debug = False
    ):
        assert type(fpmodule) == torch.nn.Conv2d
        super(LSQPlusConv2d, self).__init__(
            in_channels=fpmodule.in_channels,
            out_channels=fpmodule.out_channels,
            kernel_size=fpmodule.kernel_size,
            stride=fpmodule.stride,
            padding=fpmodule.padding,
            dilation=fpmodule.dilation,
            groups=fpmodule.groups,
        )
        self.weight = fpmodule.weight
        self.bias = fpmodule.bias
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.quant_activation = quant_activation
        self.per_channel = per_channel
        if self.per_channel:
            self.weight_alpha = Parameter(torch.Tensor(self.out_channels, 1, 1, 1))
        else:
            self.weight_alpha = Parameter(torch.Tensor(1))
        self.register_buffer("weight_init_state", torch.zeros(1))
        self.signed = signed
        self.act_alpha = Parameter(torch.Tensor(1))
        self.register_buffer("act_init_state", torch.zeros(1))
        self.add_offset = add_offset
        
        if self.per_channel == per_channel:
            self.weight_offset = Parameter(torch.Tensor(self.out_channels, 1, 1, 1))
            self.act_offset = Parameter(torch.Tensor(1, self.in_channels, 1, 1))
        else:
            self.weight_offset = Parameter(torch.Tensor(1))
            self.act_offset = Parameter(torch.Tensor(1))
        self.debug = debug

    def lsq_quantization_act(self, x):
        assert self.act_alpha is not None
        if self.signed:
            Qn = -(2 ** (self.nbits_a - 1))
            Qp = 2 ** (self.nbits_a - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits_a - 1
        if self.training and self.act_init_state == 0:
            self.act_alpha.data.copy_((x.max() - x.min()) / (Qp - Qn) * 0.9)
            if self.add_offset:
                self.act_offset.data.copy_(x.min() * 0.9 - Qn * self.act_alpha)
                #self.act_offset.data.copy_(x.mean()) 
            self.act_init_state.fill_(1)

        x_q = LSQPlus.apply(x, self.act_alpha, self.act_offset, Qn, Qp, 
                            "a",
                            self.act_alpha.numel() > 1,
                            self.act_offset.numel() > 1)
        return x_q

    def lsq_quantization_weight(self, weight):
        Qn = -(2 ** (self.nbits_w - 1))
        Qp = 2 ** (self.nbits_w - 1) - 1
            
        if self.training and self.weight_init_state == 0:
            self.weight_alpha.data.copy_(torch.max( torch.abs(weight.mean() - 3 * weight.std()), 
                                    torch.abs(weight.mean() + 3 * weight.std())) / ((Qp - Qn)/2))
            if self.add_offset:
                self.weight_offset.data.copy_(weight.mean())
            self.weight_init_state.fill_(1)

        '''
        g = 1.0 / math.sqrt(weight.numel() * Qp) if self.scale_grad else 1.0
        weight_alpha = grad_scale(self.weight_alpha, g) if self.training else self.running_weight_alpha 
        if not self.add_offset:
            w_q = round_pass((weight / weight_alpha).clamp(Qn, Qp)) * weight_alpha
        else:
            weight_offset = grad_scale(self.weight_offset, g) if self.training else self.running_weight_offset
            w_q = round_pass((weight - weight_offset) / weight_alpha).clamp(Qn, Qp) * weight_alpha + weight_offset
        '''
        w_q = LSQPlus.apply(weight, self.weight_alpha, self.weight_offset, Qn, Qp,
                            "w",
                            self.weight_alpha.numel() > 1,
                            self.weight_offset.numel() > 1)
        return w_q

    def forward(self, x):
        # if self.alpha is None:
        #     return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        w_q = self.lsq_quantization_weight(self.weight)
    
        if self.quant_activation:
            x_q = self.lsq_quantization_act(x)    
        else: 
            x_q = x
        if self.debug:
            self.Qweight = w_q.clone().detach()
            self.Qactivation = x_q.clone().detach()
        return self._conv_forward(x_q, w_q, self.bias)


class LSQPlusConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        fpmodule,
        nbits_w=8,
        nbits_a=8,
        signed=False,
        quant_activation = True,
        add_offset = True,
        per_channel=True,
        debug = False
    ):
        assert type(fpmodule) == torch.nn.ConvTranspose2d
        super(LSQPlusConvTranspose2d, self).__init__(
            in_channels=fpmodule.in_channels,
            out_channels=fpmodule.out_channels,
            kernel_size=fpmodule.kernel_size,
            stride=fpmodule.stride,
            padding=fpmodule.padding,
            output_padding=fpmodule.output_padding,
            dilation=fpmodule.dilation,
            groups=fpmodule.groups
        )
        self.weight = fpmodule.weight
        self.bias = fpmodule.bias
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.quant_activation = quant_activation
        self.per_channel = per_channel
        if self.per_channel:
            self.weight_alpha = Parameter(torch.Tensor(self.out_channels, 1, 1, 1))
        else:
            self.weight_alpha = Parameter(torch.Tensor(1))
        self.register_buffer("weight_init_state", torch.zeros(1))
        self.signed = signed
        self.act_alpha = Parameter(torch.Tensor(1))
        self.register_buffer("act_init_state", torch.zeros(1))
        self.add_offset = add_offset
        
        if self.per_channel:
            self.weight_offset = Parameter(torch.Tensor(self.out_channels, 1, 1, 1))
            self.act_offset = Parameter(torch.Tensor(1, self.in_channels, 1, 1))
        else:
            self.weight_offset = Parameter(torch.Tensor(1))
            self.act_offset = Parameter(torch.Tensor(1))
        self.debug = debug

    def lsq_quantization_act(self, x):
        assert self.act_alpha is not None
        if self.signed:
            Qn = -(2 ** (self.nbits_a - 1))
            Qp = 2 ** (self.nbits_a - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits_a - 1
        if self.training and self.act_init_state == 0:
            self.act_alpha.data.copy_((x.max() - x.min()) / (Qp - Qn) * 0.9)
            if self.add_offset:
                self.act_offset.data.copy_(x.min() * 0.9 - Qn * self.act_alpha)
                #self.act_offset.data.copy_(x.mean()) 
            self.act_init_state.fill_(1)

    
        x_q = LSQPlus.apply(x, self.act_alpha, self.act_offset, Qn, Qp,
                            "a",
                            self.act_alpha.numel() > 1,
                            self.act_offset.numel() > 1)
        return x_q

    def lsq_quantization_weight(self, weight):
        Qn = -(2 ** (self.nbits_w - 1))
        Qp = 2 ** (self.nbits_w - 1) - 1
            
        if self.training and self.weight_init_state == 0:
            self.weight_alpha.data.copy_(torch.max( torch.abs(weight.mean() - 3 * weight.std()), 
                                    torch.abs(weight.mean() + 3 * weight.std())) / ((Qp - Qn)/2))
            if self.add_offset:
                self.weight_offset.data.copy_(weight.mean())
            self.weight_init_state.fill_(1)

        '''
        g = 1.0 / math.sqrt(weight.numel() * Qp) if self.scale_grad else 1.0
        weight_alpha = grad_scale(self.weight_alpha, g) if self.training else self.running_weight_alpha 
        if not self.add_offset:
            w_q = round_pass((weight / weight_alpha).clamp(Qn, Qp)) * weight_alpha
        else:
            weight_offset = grad_scale(self.weight_offset, g) if self.training else self.running_weight_offset
            w_q = round_pass((weight - weight_offset) / weight_alpha).clamp(Qn, Qp) * weight_alpha + weight_offset
        '''
        w_q = LSQPlus.apply(weight.permute(1, 0, 2, 3), self.weight_alpha, self.weight_offset,
                            Qn, Qp,
                            "w",
                            self.weight_alpha.numel() > 1,
                            self.weight_offset.numel() > 1)
        return w_q.permute(1, 0, 2, 3)

    def forward(self, x):
        # if self.alpha is None:
        #     return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        w_q = self.lsq_quantization_weight(self.weight)
    
        if self.quant_activation:
            x_q = self.lsq_quantization_act(x)    
        else: 
            x_q = x
        if self.debug:
            self.Qweight = w_q.clone().detach()
            self.Qactivation = x_q.clone().detach()
        return F.conv_transpose2d(x_q, w_q, self.bias, self.stride, self.padding, self.output_padding)