import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

class roundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # input.data = torch.round(input.data).to(torch.int32).to(torch.float32) ###此语法在returen后会改变self.bias的值
        input = torch.round(input).to(torch.int32).to(torch.float32) ###此语法在returen后不会改变self.bias的值,because return is Tensor but not Parameter.
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        # print("roundSTE_grad: ", grad_output)
        return grad_output, None, None, None

class floorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = torch.floor(input).to(torch.int32).to(torch.float32)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        # print("floorSTE_grad: ", grad_output)
        return grad_output, None, None, None

class wSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input_out = torch.zeros_like(input) ###此语句在returen后不会改变self.weight的值(目前训练的最好的那个没有这句话)
        idx_p = (input >= 0)
        mtx_p = copy.deepcopy(input[idx_p])
        idx_n = (input < 0)
        mtx_n = copy.deepcopy(input[idx_n])
        input_out[idx_p] = e2t_p(t2e_p(mtx_p)).to(torch.int32).to(torch.float32)
        input_out[idx_n] = e2t_n(t2e_n(mtx_n)).to(torch.int32).to(torch.float32)
        return input_out ###return是tensor不会改变self.weight的值，但如果还是Parameter会连接在一起, but grad could be passed.

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        # print("wSTE_grad")
        return grad_output, None, None, None

def t2e_p(mtx):
    return (torch.round(2 ** (torch.log2(mtx+1) * 7 / 9))-1).clamp_(-128, 127)

def e2t_p(mtx):
    return torch.round(2 ** (torch.log2(mtx+1) * 9 / 7))-1

def t2e_n(mtx):
    return (-torch.round(2 ** (torch.log2(-mtx) * 7 / 9))).clamp_(-128, 127)

def e2t_n(mtx):
    return -torch.round(2 ** (torch.log2(-mtx) * 9 / 7))

class pchannel_intBit_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Bit=16):
        input_out = torch.zeros_like(input)
        for i in range(input_out.shape[0]):
            n = 2 ** (Bit - 1)
            scl = 1. / input[i].abs().max() * n
            input_out[i] = torch.clip(torch.round(input[i] * scl), -n, n-1) / scl
        return input_out
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        # print("wint16STE_grad")
        return grad_output, None, None, None

class intBit_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Bit=8):
        input_out = torch.zeros_like(input)
        n = 2 ** (Bit - 1)
        scl = 1. / input.abs().max() * n
        input_out = torch.clip(torch.round(input * scl), -n, n-1) / scl
        return input_out
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        # print("wint16STE_grad")
        return grad_output, None, None, None

class actPACT_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, val, Bit=8, if_intclp=False, clp_k=None):
        output = torch.zeros_like(input)
        n = 2 ** Bit - 1
        if if_intclp:
            scl = torch.round(1. / val * n * (2 ** 16))
            ctx.save_for_backward(input, torch.round(val * (2 ** clp_k)))
            output = torch.clip(input, 0, torch.round(val * (2 ** clp_k)))
            output = floorSTE.apply((output * scl[0] + 2 ** (15 + clp_k))/2 ** (16 + clp_k))
        else:
            scl = 1. / val * n
            ctx.save_for_backward(input, val)
            output = torch.clip(input, 0, val)
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        input, val = ctx.saved_tensors
        input_range = ~((input<0)|(input>val))
        val_range = input>val
        val_grad = torch.sum(grad_output * val_range)
        return grad_output * input_range, val_grad, None, None

class actPACT(nn.Module):
    def __init__(self, ini_val, num_bits=8, if_intclp=True, clp_k=None):
        super(actPACT, self).__init__()
        self.num_bits = num_bits
        self.learned_val = Parameter(torch.Tensor([ini_val]))
        self.if_intclp = if_intclp
        self.clp_k = clp_k

    def forward(self, input):
        input = actPACT_STE.apply(input, self.learned_val, self.num_bits, self.if_intclp, self.clp_k)
        return input