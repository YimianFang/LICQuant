import torch
LSQmodel = torch.load("checkpoint/LSQ/checkpoint_best_loss.pth.tar")
yKLmodel = torch.load("checkpoint/LSQ_yKL/checkpoint_best_loss_4_encoder.pth.tar")
LSQstate = LSQmodel['state_dict']
yKLstate = yKLmodel['state_dict']
for k, v in LSQstate.items():
    if "_fn.s" in k:
        gap = LSQstate[k].cpu() - yKLstate["module." + k].cpu()
        total = v.numel()
        print(k, "- LSQ > yKL:", round((sum(gap > 0) / total * 100).item(), 2), "%")