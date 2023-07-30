import torch
LSQmodel = torch.load("checkpoint/LSQ/checkpoint_best_loss.pth.tar")
yKLmodel = torch.load("checkpoint/LSQ_yKL/checkpoint_best_loss_4_encoder.pth.tar")
LSQstate = LSQmodel['state_dict']
yKLstate = yKLmodel['state_dict']
for k, v in LSQstate.items():
    if (".gamma" in k or ".beta" in k) and not ".beta_" in k:
        if "gamma" in k:
            print("=================", k, "=================")
            C = v.shape[0]
            for i in range(C):
                print("----------- Channel", i, "-----------")
                print(v[i].abs().topk(20)[0])