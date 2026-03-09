import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import argparse
from typing import Iterable, Optional

import wandb
from tqdm import trange
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from data import *
from model import EmbMIL
from util import AverageMeter, cat_acc

def get_args_parser():
    parser = argparse.ArgumentParser('Supervised ABMIL Panc Training', add_help=False)

    parser.add_argument('--seed',               type=int,   default=0)
    
    parser.add_argument('--model',              type=str,   default="uni")
    
    parser.add_argument('--batch_size',         type=int,   default=12)
    parser.add_argument('--loss_fn',            type=str,   default="mse")
    # parser.add_argument('--data_path',          type=str,   default="../uni_embs")
    parser.add_argument('--epochs',             type=int,   default=200)
    parser.add_argument('--device',                         default='cuda')
    parser.add_argument('--float16',            type=bool,  default=True)
    
    parser.add_argument('--prefetch_factor',    type=int,   default=2)
    parser.add_argument('--num_workers',        type=int,   default=1)
    parser.add_argument('--pin_mem',            type=bool,  default=True)
    parser.add_argument('--max_instances',      type=int,   default=5000)
    parser.add_argument('--train_split',        type=float, default=.85)

    parser.add_argument('--label_col',          type=str,   default="survival_months")
    parser.add_argument('--censor_col',         type=str,   default="survival_censor")

    # Optimizer parameters
    parser.add_argument('--weight_decay',       type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr',                 type=float, default=1e-4,   metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr',             type=float, default=1e-7,   metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs',      type=int, default=25,       metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--warmup_start',       type=float, default=1e-2,   metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument('--save_path', default="../weights",
                        help="directory to which the pretrained model weights should be saved")

    parser.add_argument('--disable_wandb', action="store_true")
    return parser

# --------------------------------------------------------

def get_opt_and_sched(model, args):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    warmup_scheduler = LinearLR(optimizer, start_factor=args.warmup_start, end_factor=1.0, total_iters=args.warmup_epochs)
    #change this later to update on each batch rather than each epoch
    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])

    return optimizer, scheduler

# --------------------------------------------------------

def train_one_epoch(model: torch.nn.Module,
                    train_loader: Iterable, 
                    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler,
                    scaler: torch.amp.grad_scaler.GradScaler,
                    device: str,
                    args=None):
    model.train(True)
    optimizer.zero_grad()

    metrics = {"Train Loss": AverageMeter(), "lr": AverageMeter()}
    # "Train Acc": AverageMeter()

    fns = {"Train MSE": F.mse_loss, "Train L1": F.l1_loss}
    test_metrics = {name: AverageMeter() for name in fns}
    metrics = {**metrics, **test_metrics}
    
    for batch in train_loader:
        bag, labels = batch
        for sample, label in zip(bag, labels):
            label = torch.tensor([label]).float().unsqueeze(0)
            label = label.to(device)
            sample = sample.unsqueeze(0)
            sample = sample.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.float16):
                loss, preds = model(h=sample, labels=label, return_attention = False)
                with torch.inference_mode():
                    for name, fn in fns.items():
                        train_loss = fn(preds, label)
                        metrics[name].update(train_loss.detach().item())

            # if args.float16:
            #     scaler.scale(loss).backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            loss.backward()
            optimizer.step()

            metrics["Train Loss"].update(loss.detach().item())

            lr = optimizer.param_groups[0]["lr"]
            metrics["lr"].update(lr)

            optimizer.zero_grad() # set_to_none=True here can modestly improve performance
        
    scheduler.step()

    return {k: meter.avg for k, meter in metrics.items()}

def test(model: torch.nn.Module, data_loader: Iterable, device: str, args=None):

    model.eval()

    # metrics = {"Test Loss": AverageMeter(), }
    # "Test Acc": AverageMeter()
    fns = {"Test MSE": F.mse_loss, "Test L1": F.l1_loss}
    metrics = {name: AverageMeter() for name in fns}

    for batch in data_loader:
        bag, labels = batch
        for sample, label in zip(bag, labels):
            label = torch.tensor([label]).float().unsqueeze(0)
            label = label.to(device)
            sample = sample.unsqueeze(0)
            sample = sample.to(device)
            with torch.inference_mode():
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.float16):
                    _, preds = model(h=sample, labels=label, return_attention = False)

                for (_, fn), (_, meter) in zip(fns.items(), metrics.items()):
                    loss = fn(preds, label)
                    meter.update(loss.detach().item())
            # acc = cat_acc(pred, label).detach().item()
            # metrics["Test Acc"].update(acc)
        
    return {k: meter.avg for k, meter in metrics.items()}

# --------------------------------------------------------

def get_loaders(args):
    index = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)
    inds = np.arange(len(index[args.label_col]))
    filter = (index[args.censor_col] == 0) & (~np.isnan(index[args.label_col]))

    valid_inds = inds[filter]
    num_train = int(len(valid_inds) * args.train_split)
    num_valid = len(valid_inds) - num_train
    print(f"Found: {len(valid_inds)} valid samples split into {num_train} train and {num_valid} validation")
    
    del index, inds, filter

    np.random.shuffle(valid_inds)
    train_inds, test_inds = valid_inds[:num_train], valid_inds[num_train:]

    dataset_args = {
        "data_dir": args.data_path,
        "max_instances": args.max_instances,
        "return_key": False,
        "label_column": args.label_col
    }
    loader_args = {
        "batch_size": args.batch_size,
        "pin_memory": args.pin_mem,
        "num_workers": args.num_workers,
        "collate_fn": collate_bags,
        "persistent_workers": args.num_workers > 0,
        "drop_last": False,
    }

    train_set = MemmapDataset(indices=train_inds, **dataset_args)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    test_set = MemmapDataset(indices=test_inds, **dataset_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    return train_loader, test_loader

# --------------------------------------------------------

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    args.data_path = f"../{args.model}_embs"

    train_loader, test_loader = get_loaders(args)
    print(f"Train Samples: {len(train_loader)}\n"
          f"Test Samples: {len(test_loader)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    losses = {
        "l1": F.l1_loss,
        "smooth_l1": F.smooth_l1_loss,
        "mse": F.mse_loss
    }

    model = EmbMIL(loss_fn=losses[args.loss_fn])
    model = model.to(device)

    optimizer, scheduler = get_opt_and_sched(model, args)
    scaler = torch.amp.GradScaler(device, enabled=True)

    if args.disable_wandb: run = None
    else:
        config = {
            "Model": args.model,
            # "lr": args.lr,
            # **loader_args
            "Loss": args.loss_fn,
            "Seed": args.seed
        }


        run = wandb.init(
            entity="bumjin_joo-brown-university", 
            project=f"Panc MM", 
            name=f"{args.model} ABMIL - {args.label_col}", 
            config=config
        )

    print(f"Start training for {args.epochs} epochs")
    pbar = trange(0, args.epochs, desc="Training Epochs", postfix={})
    for _ in pbar:
        train_stats = train_one_epoch(model, train_loader,
                                      optimizer, scheduler,
                                      scaler,
                                      device,
                                      args)
        test_stats = test(model, test_loader, device, args=args)

        postfix = {**train_stats, **test_stats}
        if run is not None: run.log(postfix)
        pbar.set_postfix(postfix)
       

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)