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

from MixQ_model import MixQHyperprior
from compressai.zoo import models


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
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
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out
    
class MSELoss_y(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output):
        out = {}
        out["loss"] = self.mse(output["y"], output["yy"])

        return out

class SumLoss_y(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()

    def forward(self, output):
        out = {}
        out["loss"] = (output["y"] - output["yy"]).abs().sum()

        return out

class MeanLoss_y(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()

    def forward(self, output):
        out = {}
        out["loss"] = (output["y"] - output["yy"]).abs().mean()

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

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
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
    model,  optimizer, epoch, clip_max_norm, save
):
    global best_loss
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        # aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # aux_loss = model.aux_loss()
        # aux_loss.backward()
        # aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.6f} |'
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
                        # "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    i
                )


def test_epoch(epoch, model, Q=False):
    model.eval()
    device = next(model.parameters()).device

    Meanloss = AverageMeter()
    different_qy = AverageMeter()
    total_qy = AverageMeter()
    RDloss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d, Q)
            out_criterion = criterion(out_net)
            out_criterionRD = criterionRD(out_net, d)

            Meanloss.update(out_criterion["loss"])
            different_qy.update((out_net["y"].round() != out_net["yy"].round()).sum())
            total_qy.update(out_net["y"].numel())
            RDloss.update(out_criterionRD["loss"])
            mse_loss.update(out_criterionRD["mse_loss"])
            bpp_loss.update(out_criterionRD["bpp_loss"])
            aux_loss.update(model.aux_loss())

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tMean Loss: {Meanloss.avg:.5f} |"
        f"\tPercent of different qy: {different_qy.avg:.1f}/{total_qy.avg:.1f} = {different_qy.avg / total_qy.avg * 100:.4f} %"
    )

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tRD Loss: {RDloss.avg:.5f} |"
        f'\tPSNR: {cntPSNR(mse_loss.avg):.3f} |'
        f"\tMSE loss: {mse_loss.avg:.5f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}"
    )

    return Meanloss.avg, RDloss.avg


def save_checkpoint(state, is_best, step):
    root = "./checkpoint/" + subroot
    if not os.path.exists(root):
        os.mkdir(root)
    filename = os.path.join(root, "epoch_{}_step_{}_loss_{:.6f}.pth.tar".format(state["epoch"], step, state["loss"]))
    torch.save(state, filename)
    if is_best:
        print("New Best Checkpoint Saved!")
        shutil.copyfile(filename, os.path.join(root, "checkpoint_best_loss.pth.tar"))
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
        "-ql", "--QLayerList", nargs='+', type=int, required=True, help="The convolution layers quantized"
    )
    parser.add_argument(
        "--default_precision", type=int, default=8, help="The default precision for quantization"
    )
    parser.add_argument(
        "--mixed_precision", type=int, default=16, help="The mixed precision for quantization"
    )
    parser.add_argument(
        "-cnum", "--mixed_channels_num", type=int, default=1, help="The num of channels quantized with mixed precision"
    )
    parser.add_argument(
        "--absn", dest="abs", action="store_true", required=True, help="Sort the channels with abs n of noise"
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
        type=float,
        default=0.0250,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: %(default)s)"
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

    global train_dataloader, test_dataloader, criterion, lr_scheduler, best_loss, subroot, QLayerList, criterionRD
    subroot = args.subroot
    QLayerList = args.QLayerList
    default_precision = args.default_precision
    mixed_precision = args.mixed_precision
    top_k = args.mixed_channels_num
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
    net = MixQHyperprior(prefpnet, default_precision=default_precision, mixed_precision=mixed_precision)
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

    # optimizer, aux_optimizer = configure_optimizers(net, args)
    parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".UNoise_n") and str(args.NoiseLayerNum) in n and p.requires_grad
    }

    # print("===The Parameters to Update===")
    # print(parameters)
    # params_dict = dict(net.named_parameters())
    # optimizer = optim.Adam(
    #     (params_dict[n] for n in sorted(parameters)),
    #     lr=args.learning_rate,
    # )
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = MeanLoss_y()
    criterionRD = RateDistortionLoss(args.lmbda)

    # print("===The Performance of The Pre-trained Floating Point Model===")
    # test_epoch(0, prefpnet.to(device))

    last_epoch = 0
    best_loss = float("inf")
    if args.checkpoint:  # load from previous quantized checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        net.load_state_dict(checkpoint["state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print("===The Performance of The Quantized Checkpoint===")
        test_epoch(last_epoch, net)

    Meantop, RDtop = [], []
    Meanbottom, RDbottom = [], []
    Meanrand, RDrand = [], []
    print("===To Execute Mix-precision Quantization===")
    UNstate = torch.load("checkpoint/UniformN/checkpoint_layer6_best_loss.pth.tar", map_location=device)["state_dict"]
    N_n = {}
    for l in QLayerList:
        N_n[l] = UNstate["g_a_noise." + str(l) + ".UNoise_n"]
    print("default_precision:", default_precision)
    print("mixed_precision:", mixed_precision)
    print("===Default precision Quantization Inference===")
    loss = test_epoch(0, net, False)
    print("===Mix-precision Quantization Inference===")
    for k in range(1, top_k + 1, 5):
        print("== Num of channels quantized with mixed precision:", k, "==")
        print("== Top k ==")
        mixed_channels = {}
        for l, v in N_n.items():
            if args.abs:
                mixed_channels[l] = torch.topk(v.abs(), k)[1].tolist() # abs
            else:
                mixed_channels[l] = torch.topk(v, k)[1].tolist()
        net.g_a_layer_init_scale(QLayerList, mixed_channels=mixed_channels)
        Meanloss, RDloss = test_epoch(0, net, True)
        Meantop.append(Meanloss.item())
        RDtop.append(RDloss.item())
        print("== Bottom k ==")
        mixed_channels = {}
        for l, v in N_n.items():
            if args.abs:
                mixed_channels[l] = torch.topk(v.abs(), k, largest=False)[1].tolist() # abs
            else:
                mixed_channels[l] = torch.topk(v, k, largest=False)[1].tolist()
        net.g_a_layer_init_scale(QLayerList, mixed_channels=mixed_channels)
        Meanloss, RDloss = test_epoch(0, net, True)
        Meanbottom.append(Meanloss.item())
        RDbottom.append(RDloss.item())
        print("== Random k ==")
        mixed_channels = {}
        for l, v in N_n.items():
            mixed_channels[l] = torch.randint(0, v.numel(), (k,)).tolist()
        net.g_a_layer_init_scale(QLayerList, mixed_channels=mixed_channels)
        Meanloss, RDloss = test_epoch(0, net, True)
        Meanrand.append(Meanloss.item())
        RDrand.append(RDloss.item())
        print("\n")
    filename = "result_abs" if args.abs else "result"
    torch.save({"Meantop": Meantop, "Meanbottom": Meanbottom, "Meanrand": Meanrand,
                "RDtop": RDtop, "RDbottom": RDbottom, "RDrand": RDrand,}, "MixQ/" + filename + "/layer" + str(QLayerList) + ".pth.tar")
    # for epoch in range(last_epoch, args.epochs):
    #     print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    #     train_one_epoch(
    #         net,
    #         optimizer,
    #         epoch,
    #         args.clip_max_norm,
    #         args.save
    #     )
    #     loss = test_epoch(epoch, net)
    #     lr_scheduler.step(loss)

    #     is_best = loss < best_loss
    #     best_loss = min(loss, best_loss)

    #     if args.save:
    #         save_checkpoint(
    #             {
    #                 "epoch": epoch,
    #                 "state_dict": net.state_dict(),
    #                 "loss": loss,
    #                 "optimizer": optimizer.state_dict(),
    #                 "lr_scheduler": lr_scheduler.state_dict(),
    #             },
    #             is_best,
    #             len(train_dataloader),
    #             args.mehtod
    #         )


if __name__ == "__main__":
    main(sys.argv[1:])