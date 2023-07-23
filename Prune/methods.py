import torch

def Pruning1(qtensor, fptensor):
    shape = qtensor.shape
    result = torch.zeros(qtensor.numel())
    kldivloss = torch.nn.KLDivLoss()
    for i in range(qtensor.numel()):
        q_take = torch.index_select(qtensor, i)
        fp_take = torch.index_select(fptensor, i)
        result[i] = kldivloss(q_take, fp_take)
    return result.view(shape)