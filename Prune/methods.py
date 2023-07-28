import torch
import copy

def Pruning1(qtensor, fptensor):
    _, C, H, W = qtensor.shape
    result = torch.zeros(C)
    kldivloss = torch.nn.KLDivLoss()
    for i in range(C):
        mask = torch.ones_like(qtensor)
        mask[:, C, ...].fill_(0)
        # q_take = torch.masked_select(qtensor, mask)
        # fp_take = torch.masked_select(fptensor, mask)
        q_take = qtensor * mask
        fp_take = fptensor
        result[i] = kldivloss(q_take, fp_take)
    return result.view(-1, C, H, W)

def Pruning2(qtensor):
    _, C, H, W = qtensor.shape
    maxv = qtensor.amax(dim=(0, 2, 3))
    minv = qtensor.amin(dim=(0, 2, 3))
    gapv = maxv - minv
    return gapv