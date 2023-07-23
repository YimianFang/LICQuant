import os
import argparse
import math
import random
import shutil
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder

from LSQPlus_yKL_reGDN.LSQPlus_yKL_reGDN_model import LSQPlus_reGDN_ScaleHyperprior
from compressai.zoo import models

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(log_target=True)
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["kl_loss"] = torch.abs(self.kl(output["y_hat"], output["y_fp_hat"]))
        if out["kl_loss"] * 1e-10 > 1:
            kl_lmbda = 1e-10
        else:
            kl_lmbda = 1e4
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        out["loss"] += out["kl_loss"] * kl_lmbda

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    if args.training_params == "encoder":
        parameters = {
            n
            for n, p in net.named_parameters()
            if not n.endswith(".quantiles") and "_fp" not in n  and ("g_s" not in n and "h_a" not in n and "h_s" not in n) and p.requires_grad
        }
    elif args.training_params == "decoder":
        parameters = {
            n
            for n, p in net.named_parameters()
            if not n.endswith(".quantiles") and "_fp" not in n  and ("g_a" not in n) and p.requires_grad
        }
    elif args.training_params == "e2e":
        parameters = {
            n
            for n, p in net.named_parameters()
            if not n.endswith(".quantiles") and "_fp" not in n and p.requires_grad
        }


    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    # assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model,  optimizer, aux_optimizer, epoch, clip_max_norm, save
):
    global best_loss
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3e} |'
                f'\tPSNR: {cntPSNR(out_criterion["mse_loss"].item()):.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tKL loss: {out_criterion["kl_loss"].item():.2e} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

        if i % 500 == 0:
            loss = test_epoch(epoch, model)
            model.train()
            lr_scheduler.step(loss)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    i
                )


def test_epoch(epoch, model):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    kl_loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            kl_loss.update(out_criterion["kl_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3e} |"
        f'\tPSNR: {cntPSNR(mse_loss.avg):.3f} |'
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tKL loss: {kl_loss.avg:.2e} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}"
    )

    return loss.avg


def save_checkpoint(state, is_best, step):
    root = "checkpoint/" + subroot
    if not os.path.exists(root):
        os.mkdir(root)
    filename = os.path.join(root, "epoch_{}_step_{}_loss_{:.4f}.pth.tar".format(state["epoch"], step, state["loss"]))
    torch.save(state, filename)
    if is_best:
        print("New Best Checkpoint Saved!")
        shutil.copyfile(filename, os.path.join(root, "checkpoint_best_loss_" + cp_order + "_" + training_params + ".pth.tar"))
    print("\n")

def cntPSNR(mse):
    if type(mse) == torch.Tensor:
        mse = mse.clone().detach()
    else:
        mse = torch.tensor(mse)
    psnr = 10 * (torch.log(1 * 1 / mse) / torch.log(torch.tensor(10.)))
    return psnr

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-sr", "--subroot", type=str, required=True, help="the subroot path to save checkpoints"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-tp", "--training_params", type=str, required=True, help="the parameters to train"
    )
    parser.add_argument(
        "-ord", "--cp_order", type=str, required=True, help="the order of checkpoint"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        required=True,
        default=5,
        help="The quality of pretrained floating point model",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        required=True, 
        type=float,
        default=0.0250,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("-id", "--cuda_id", type=int)
    parser.add_argument(
        "--save", action="store_true", help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a quantized checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and args.cuda_id:
        torch.cuda.set_device(args.cuda_id)
        device = device + ':' + str(args.cuda_id)

    global train_dataloader, test_dataloader, criterion, lr_scheduler, best_loss, subroot, training_params, cp_order
    subroot = args.subroot
    training_params = args.training_params
    cp_order = args.cp_order
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.cuda,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.cuda,
    )

    prefpnet = models["bmshj2018-hyperprior"](quality=args.quality, metric="mse", pretrained=True) # quality=5, lambda=0.0250, N=128, M=192
    net = LSQPlus_reGDN_ScaleHyperprior(prefpnet)
    # fpstate = {}
    # for k in net.state_dict():
    #     if k in prefpnet.state_dict():
    #         v = prefpnet.state_dict()[k]
    #     else:
    #         v = net.state_dict()[k]
    #     fpstate[k] = v
    # net.load_state_dict(fpstate)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1 and not args.cuda_id:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    # print("===The Performance of The Pre-trained Floating Point Model===")
    # test_epoch(0, prefpnet.to(device))

    last_epoch = 0
    best_loss = float("inf")
    if args.checkpoint:  # load from previous quantized checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"]
        if cp_order: best_loss = checkpoint["loss"]
        prestate = checkpoint["state_dict"]
        newstate = net.state_dict()
        state = {}
        for k, v in newstate.items():
            if k.replace("module.", "") not in prestate: state[k] = v
            else: state[k] = prestate[k.replace("module.", "")]
        net.load_state_dict(state)
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print("===The Performance of The Quantized Checkpoint===")
        test_epoch(last_epoch, net)

    
    print("===QAT Begins===")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            args.save
        )
        loss = test_epoch(epoch, net)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                len(train_dataloader)
            )


if __name__ == "__main__":
    main(sys.argv[1:])