import copy

from torch.autograd import Variable
import torch
from torch import nn
from collections import OrderedDict
import math
import numpy as np

def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * (len(sorted_value)/2))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = v.data.cpu().numpy()
    #sf = math.ceil(math.log2(v+1e-12))
    sf=v
    return sf

def get_sf_CW(model, state_dict, overflow_rate=0):
    sf_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            para = state_dict[(name + '.weight').replace('module.','')]
            abs_value = para.abs().view(para.shape[0], -1)
            sf = []
            for i in range(para.shape[0]):
                sorted_value = abs_value[i].sort(dim=0, descending=True)[0]
                split_idx = int(overflow_rate * (len(sorted_value) / 2))
                v = sorted_value[split_idx]
                if isinstance(v, Variable):
                    v = v.data.cpu().numpy()
                sf.append(v)
            sf_list.append(sf)
    return sf_list

def changemodelbit_Waseda_CW(Bits,model,sf_list,state):
    state_dict_quant = model.state_dict()
    kk = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            v = state[(name + '.weight').replace('module.','')]
            if isinstance(Bits, int):
                bits = Bits
            else:
                bits = Bits[kk]
            v_quant = Waseda_quantize_CW(v, sf_list[kk], bits=bits)
            kk = kk + 1
            state_dict_quant[name + '.weight'] = v_quant
    model.load_state_dict(state_dict_quant)

def changemodelbit_CW(Bits,model,sf_list,state):
    state_dict_quant = model.state_dict()
    kk = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            v = state[(name + '.weight').replace('module.','')]
            if isinstance(Bits, int):
                bits = Bits
            else:
                bits = Bits[kk]
            v_quant = linear_quantize_CW(v, sf_list[kk], bits=bits)
            kk = kk + 1
            state_dict_quant[name + '.weight'] = v_quant
    model.load_state_dict(state_dict_quant)

def changemodelbit_intconv(Bits,model,sf_list,state):
    state_dict_quant = model.state_dict()
    kk = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if "priorDecoder" in name:
                v = state[(name + '.weight').replace('module.','')]
                if isinstance(Bits, int):
                    bits = Bits
                else:
                    bits = Bits[kk]
                v_quant = linear_intconv_CW(v, sf_list[kk], bits=bits)
                state_dict_quant[name + '.weight'] = v_quant
                print((v - state_dict_quant[name + '.weight']).abs().max())
            kk = kk + 1
    model.load_state_dict(state_dict_quant)

def Waseda_quantize_CW(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return (torch.sign(input) - 0.5) * sf
    # delta = math.pow(2.0, -sf)
    FL = bits - 2
    clipped_value = torch.zeros_like(input)
    for i in range(input.shape[0]):
        sfk = torch.pow(2.0, -torch.floor(torch.log2(torch.Tensor(sf[i]))))
        swk = input[i] * sfk
        clipped_value[i, (swk >= 0.5) & (swk < 2)] = torch.round(swk[(swk >= 0.5) & (swk < 2)] * math.pow(2.0, FL - 1)) / math.pow(2.0, FL - 1) / sfk
        clipped_value[i, (swk >= 0.25) & (swk < 0.5)] = torch.round(swk[(swk >= 0.25) & (swk < 0.5)] * math.pow(2.0, FL)) / math.pow(2.0, FL) / sfk
        clipped_value[i, swk < 0.25] = torch.round(swk[swk < 0.25] * math.pow(2.0, FL + 2)) / math.pow(2.0, FL + 2) / sfk
    return clipped_value

def intconvq_CW_0(weight, bias):
    round_weight = torch.zeros_like(weight)
    round_bias = torch.zeros_like(torch.Tensor(bias.shape[0]))
    scale = torch.zeros_like(torch.Tensor(weight.shape[0]))
    for i in range(weight.shape[0]):
        scale[i] = 100000
        round_weight[i] = torch.round(weight[i] * scale[i])
        round_bias[i] = torch.round(bias[i] * scale[i])
    return round_weight, round_bias, scale

def intconvq_CW_1(bias, relu_scale = None):
    # print(i, weight[i].max(), weight[i].min())
    copy_bias = copy.deepcopy(bias)
    copy_bias *= relu_scale
    # print(round_weight[i].abs().max())
    # print(round_bias[i].abs().max())
    return copy_bias

def intconvdeq_CW_0(input, scale):
    input = input
    deround_input = torch.zeros_like(input)
    for i in range(input.shape[0]):
        deround_input[i] = input[i] / scale[i]
    return deround_input

def intconvdeq_CW_1(input, scale):
    input = input
    scale = scale.detach()
    # print(len(input.shape))
    if len(input.shape) == 4:
        for i in range(input.shape[1]):
            input[:, i, :, :] /= scale[i]
    elif len(input.shape) == 3:
        for i in range(input.shape[0]):
            input[i] /= scale[i]
    return input

def linear_quantize_CW(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return (torch.sign(input) - 0.5)*sf
    #delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    clipped_value = torch.zeros_like(input)
    for i in range(input.shape[0]):
        delta = sf[i] / bound
        rounded = torch.round(input[i] / delta)
        clipped_value[i] = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

def linear_intconv_CW(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return (torch.sign(input) - 0.5)*sf
    #delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    clipped_value = torch.zeros_like(input)
    for i in range(input.shape[0]):
        delta = sf[i] / bound
        rounded = torch.round(input[i] / delta)
        clipped_value[i] = torch.clamp(rounded, min_val, max_val)
    return clipped_value

def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return (torch.sign(input) - 0.5)*sf
    #delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    delta = sf/bound
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

def quant_model_bit(model,Bits):
    state_dict_quant = model.state_dict()
    sf_list=[]
    snr_dict={}
    kk=0  
    for name, module in model.named_modules():
      if isinstance(module, nn.Conv2d):
        if isinstance(Bits,int):
            bits=Bits
        else:
            bits=Bits[kk]
        kk=kk+1
        v=state_dict_quant[name+'.weight']
        print([name,bits])
        nornow=1e10
        vq = v
        for clip in [0, 0.005,0.01,0.02]:
            sf =  compute_integral_part(v, overflow_rate=clip)
            v_quant  = linear_quantize(v, sf, bits=bits)
            norv=(v_quant-v).norm()
            if nornow > norv:
                vq = v_quant
                nornow = norv
                sf_t=sf
        sf_list.append(sf_t)
        state_dict_quant[name+'.weight'] = vq
        error_noise=torch.norm(vq-v)
        par_power=torch.norm(v)
        snr=error_noise/par_power
        snr_dict[name]=snr
    model.load_state_dict(state_dict_quant)
    return model, sf_list, snr_dict

def changemodelbit_fast(Bits,model,sf_list,state):
    state_dict_quant = model.state_dict()
    kk=0
    for name, module in model.named_modules():
      if isinstance(module, nn.Conv2d):
        v=state[name[7:]+'.weight']
        if isinstance(Bits,int):
            bits=Bits
        else:
            bits=Bits[kk]
        sf = 6. if sf_list is None else sf_list[kk]  #quant.compute_integral_part(v, overflow_rate=clip)
        v_quant  = linear_quantize(v, sf, bits=bits)
        kk=kk+1
        state_dict_quant[name+'.weight'] = v_quant
    model.load_state_dict(state_dict_quant)
 
def load_state(model,state_dict):
    for name, para in model.named_parameters():
        if  'max' in name:
            continue
        state_dict[name[7:]] = para.data   
    
def quantback(model,state_dict):
    for name, module in model.named_modules():
      if isinstance(module, nn.Conv2d):
        # if module.weight.grad is not None:
        #   print(name, module.weight.grad.abs().max())
        state_dict[(name +'.weight').replace('module.','')].grad = module.weight.grad # name[7:]
        
def float_model(model,state_dict):
    dict = model.state_dict()
    for name, module in model.named_modules():
      if isinstance(module, nn.Conv2d):
        dict[name+'.weight'] = state_dict[name[7:]+'.weight']
    return dict      


#########################################Activation
def __PintQuantize__(num_bit, tensor, clamp_v):
    qmin = 0.
    qmax = 2. ** num_bit - 1.
    scale = clamp_v / (qmax - qmin)
    scale = torch.max(scale.to(tensor.device), torch.tensor([1e-8]).to(tensor.device))
    output = tensor.detach()
    output = output.div(scale)
    output = output.clamp(qmin, qmax).round()  # quantize
    output = output.mul(scale)  # dequantize

    return output.view(tensor.shape)


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bit, clamp):

        output = __PintQuantize__(bit, input, clamp)
        #ctx.mark_dirty(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None


class ActFn(torch.autograd.Function):#https://github.com/KwangHoonAn/PACT/blob/master/module.py
	@staticmethod
	def forward(ctx, x, alpha, k):
		ctx.save_for_backward(x, alpha)
		y = torch.clamp(x, min = 0, max = alpha.item())
		scale = (2**k - 1) / alpha
		y_q = torch.round( y * scale) / scale
		return y_q

	@staticmethod
	def backward(ctx, dLdy_q):
		x, alpha, = ctx.saved_tensors 
		lower_bound      = x < 0
		upper_bound      = x > alpha
		x_range = ~(lower_bound|upper_bound)
		grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
		return dLdy_q * x_range.float(), grad_alpha, None


class QuantReLU(nn.ReLU):
    def __init__(self, bit=8, channelnum=1):
        super(QuantReLU, self).__init__()
        self.quant_bit = bit
        self.register_buffer('statistic_max', torch.ones(1))
        self.momentum = 0.1
        self.running = False
        self.numact = 0

    def forward(self, input):
        out = nn.functional.relu(input)
        if self.running and (not self.training):
            size = input.size()
            self.numact = input.numel()
            Max = out.max()
            self.statistic_max.data = torch.max(self.statistic_max.data, Max)
        else:
            out = LinearQuantizeSTE.apply(out, self.quant_bit, self.statistic_max.data)

        return out

class QuantReLU_pact(nn.ReLU):
    def __init__(self, bit=8, channelnum=1):
        super(QuantReLU_pact, self).__init__()
        self.quant_bit = bit
        self.clip_max=nn.Parameter(torch.tensor(1.))
        self.momentum = 0.1
        self.running = False
        self.numact = 0

    def forward(self, input):
        if self.running and (not self.training):
            out = nn.functional.relu(input)
            size = input.size()
            self.numact = input.numel()
            Max = out.max()
            self.clip_max.data = torch.max(self.clip_max.data, Max)
        else:
            out = ActFn.apply(input, self.clip_max, self.quant_bit)

        return out

class DynamicQuantReLU(nn.ReLU):
    def __init__(self, bit=8, channelnum=1):
        super(DynamicQuantReLU, self).__init__()
        self.quant_bit = bit
        self.clamp = torch.FloatTensor([6.0])
        self.numact = 0

    def forward(self, input):
        out = nn.functional.relu(input)
        self.clamp = out.max()
        self.numact = max(input.numel(),self.numact)
        out = LinearQuantizeSTE.apply(out, self.quant_bit, self.clamp)

        return out


import math

c = None
cn = None
def quant_relu_module(m,n_dict,pre=''):
    global c,cn
    children = list(m.named_children())
    if  pre=='module':
        pre=''
    if pre:
        pre=pre+'.'
    else:
        c = None
        cn = None
    for name, child in children:
        if isinstance(child, nn.ReLU) or isinstance(child, nn.ReLU6):
            if c == None:
                assert False
                continue

            noir=n_dict[cn]

            bit = round(math.log2(3.)-math.log2(noir)) #8# if 'extr' in pre else round(math.log2(3)-math.log2(noir))
            bit = min(8,bit)
            m._modules[name] = QuantReLU(bit,c.out_channels)
            print((pre+name, bit))
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = pre+name
        else:
            quant_relu_module(child,n_dict,pre+name)


def quant_relu_module_d(m,dict,pre=''):
    children = list(m.named_children())
    c = None
    cn = None
    if pre:
        pre=pre+'.'

    for name, child in children:
        if (isinstance(child, nn.ReLU) or isinstance(child, nn.ReLU6)) and 'layer' in pre:
            if c == None:
                assert False
                continue

            noir=dict[pre+cn]

            bit = round(math.log2(3)-math.log2(noir)) #8# if 'extr' in pre else round(math.log2(3)-math.log2(noir))
            bit = min(8,bit)
            m._modules[name] = DynamicQuantReLU(bit,c.out_channels)
            print((pre+name, bit))
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            quant_relu_module_d(child,dict,pre+name)

def quant_relu_module_bit(m,bit,pre=''):
    children = list(m.named_children())
    c = None
    if pre:
        pre=pre+'.'

    for name, child in children:
        if (isinstance(child, nn.ReLU) or isinstance(child, nn.ReLU6)) and 'layer' in pre:
            if c == None:
                assert False
                continue

            m._modules[name] = QuantReLU(bit,c.out_channels)
            print((pre+name, bit))
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
        else:
            quant_relu_module_bit(child,bit,pre+name)


def dequant_relu_module(m):
    children = list(m.named_children())

    for name, child in children:
        if isinstance(child, QuantReLU) or isinstance(child, DynamicQuantReLU) or isinstance(m, QuantReLU_pact):
            m._modules[name] = nn.ReLU()
        elif isinstance(child, nn.Conv2d):
            continue
        else:
            dequant_relu_module(child)


def running_module(m):
    children = list(m.named_modules())

    for name, mc in children:
        if isinstance(mc, QuantReLU) or isinstance(mc, QuantReLU_pact):
            mc.running = True
            mc.training = False


def test_module(m):
    children = list(m.named_modules())

    for name, mc in children:
        if isinstance(mc, QuantReLU) or isinstance(mc, QuantReLU_pact):
            mc.running = False

def act_averagebit(model):
    children = list(model.named_modules())
    totalbit=0
    totalact=0

    for name, mc in children:
        if isinstance(mc, QuantReLU) or isinstance(mc, DynamicQuantReLU) or isinstance(mc, QuantReLU_pact):
            totalact += mc.numact
            totalbit += mc.numact*mc.quant_bit
    return totalbit/totalact

#******************************************
class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()
    def forward(self, x):
        return x


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True, groups=conv.groups)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

c = None
cn = None
def fuse_module(m):
    children = list(m.named_children())
    global c, cn
    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            if c==None:
                m._modules[name] = DummyModule()
                continue
            try:
                bc = fuse(c, child)
                #print('+1')
            except Exception as e:
                print(e)
                print(c)
                print(child)
                assert False
            
            m._modules[cn] = bc
            child.reset_parameters()
            child.weight.data.fill_(1.)
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)