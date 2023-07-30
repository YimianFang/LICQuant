import torch
LSQmodel = torch.load("checkpoint/LSQ/checkpoint_best_loss.pth.tar")
StaQmodel = torch.load("checkpoint/StaQ_avg/checkpoint_best_loss.pth.tar")
LSQstate = LSQmodel['state_dict']
StaQstate = StaQmodel['state_dict']
LSQscale, StaQscale = {}, {}
for k, v in StaQstate.items():
    if 'weight' in k:
        if '_a' in k or 'h_s.4' in k:
            scale = v.abs().reshape(v.shape[0], -1).max(dim=1)[0] / 2 ** 7
        elif '_s' in k:
            scale = v.abs().permute(1, 0, 2, 3).reshape(v.shape[1], -1).max(dim=1)[0] / 2 ** 7
        StaQscale[k.replace('weight', 'quan_w_fn.s')] = scale
    elif 'quan_a_fn.val' in k:
        StaQscale[k.replace('val', 's')] = v / 2 ** 7
for k, v in LSQstate.items():
    if 'quan_' in k:
        LSQscale[k] = v.view(-1)
for k, v in LSQscale.items():
    print(k, v - StaQscale[k])
