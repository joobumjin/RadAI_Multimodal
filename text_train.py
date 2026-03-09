import os
import argparse
from typing import Iterable, Optional
from collections import defaultdict

import wandb
from tqdm import trange
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from sksurv.metrics import cumulative_dynamic_auc

from data import *
from model import EmbPred
from util import *

def get_args_parser():
    parser = argparse.ArgumentParser('Supervised ABMIL Panc Training', add_help=False)

    parser.add_argument('--seed',               type=int,   default=0)
    
    parser.add_argument('--model',              type=str,   default="conch")
    parser.add_argument('--per_sample',         action="store_true")
    
    parser.add_argument('--batch_size',         type=int,   default=12)
    parser.add_argument('--loss_fn',            type=str,   default="mse")
    parser.add_argument('--data_path',          type=str,   default="../{model}_path_rad_text_embs")
    parser.add_argument('--epochs',             type=int,   default=200)
    parser.add_argument('--device',                         default='cuda')
    parser.add_argument('--float16',            type=bool,  default=True)
    parser.add_argument('--emb_merge',          type=str,   default="sum")
    
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
    parser.add_argument('--warmup_epochs',      type=int,   default=50,     metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--warmup_start',       type=float, default=1e-2,   metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument('--save_path',                      default="../weights",
                        help="directory to which the pretrained model weights should be saved")

    parser.add_argument('--disable_wandb',      action="store_true")
    parser.add_argument('--debug',              action="store_true")
    return parser

# --------------------------------------------------------

def get_opt_and_sched(model, args, iter_per_epoch = None):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    #linear warmup
    wu_iters = args.warmup_epochs * iter_per_epoch if iter_per_epoch is not None else args.warmup_epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=args.warmup_start, end_factor=1.0, total_iters=wu_iters)
    #cosine anneal 
    t_max = (args.epochs - args.warmup_epochs) * iter_per_epoch if iter_per_epoch is not None else (args.epochs - args.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.min_lr)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[wu_iters])

    return optimizer, scheduler

# --------------------------------------------------------

def get_metrics(split: str, args):
    bool_cols = set(["survival_indicator"])
    bool_var = args.label_col in bool_cols

    metrics = {f"{split} Loss": AverageMeter(), "lr": AverageMeter()}
    fns = {"Acc": lambda p, l: acc(torch.sigmoid(p) > 0.5, l)} if bool_var else {"MSE": F.mse_loss, "L1": F.l1_loss, "5yr Acc": lambda p, l: acc(p > 60, l > 60)}
    fns = {f"{split} {name}": fn for name, fn in fns.items()}
    test_metrics = {f"{name}": AverageMeter() for name in fns}
    metrics = {**metrics, **test_metrics}

    return metrics, fns

# def end_of_epoch_metrics(model: torch.nn.Module,
#                          train_loader,
#                          test_loader,
#                          args,
#                          times = [2, 5],):

    
#     aucs, avg_auc_timed = cumulative_dynamic_auc(y_train, y_test, preds, times)



def train_one_epoch(model: torch.nn.Module,
                    train_loader: Iterable, merge_fn,
                    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler,
                    scaler: torch.amp.grad_scaler.GradScaler,
                    device: str,
                    args=None):
    model.train(True)
    optimizer.zero_grad()

    metrics, fns = get_metrics("Train", args)
    
    for batch in train_loader:
        bag, labels = batch
        for sample, label in zip(bag, labels):
            label = torch.tensor([label]).float().unsqueeze(0)
            label = label.to(device)
            sample = merge_fn(sample)
            sample = sample.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.float16):
                loss, preds = model(h=sample, labels=label)
                with torch.inference_mode():
                    metrics["Train Loss"].update(loss.detach().item())
                    metrics["lr"].update(optimizer.param_groups[0]["lr"])
                    for name, fn in fns.items():
                        train_loss = fn(preds, label)
                        metrics[name].update(train_loss.detach().item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()

    return {k: meter.avg for k, meter in metrics.items()}

def train_one_batched(model: torch.nn.Module,
                    train_loader: Iterable, merge_fn,
                    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler,
                    scaler: torch.amp.grad_scaler.GradScaler,
                    device: str,
                    args=None):
    model.train(True)
    optimizer.zero_grad()

    metrics, fns = get_metrics("Train", args)
    
    for batch in train_loader:
        metrics["lr"].update(optimizer.param_groups[0]["lr"])

        bag, labels, _ = batch
        batch_loss = torch.tensor(0.0).float()
        batch_loss = batch_loss.to("cuda")
        for sample, label in zip(bag, labels):
            label = torch.tensor([label]).float().unsqueeze(0)
            label = label.to(device)
            sample = merge_fn(sample)
            sample = sample.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.float16):
                loss, preds = model(h=sample, labels=label)
                batch_loss += loss
                with torch.inference_mode():
                    metrics["Train Loss"].update(loss.detach().item())
                    for name, fn in fns.items():
                        train_loss = fn(preds, label)
                        metrics[name].update(train_loss.detach().item())

        # batch_loss /= len(labels)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return {k: meter.avg for k, meter in metrics.items()}

def test(model: torch.nn.Module, data_loader: Iterable, merge_fn, device: str, args=None):
    model.eval()

    metrics, fns = get_metrics("Test", args)

    for batch in data_loader:
        bag, labels, ids = batch
        for sample, label in zip(bag, labels):
            label = torch.tensor([label]).float().unsqueeze(0)
            label = label.to(device)
            sample = merge_fn(sample)
            sample = sample.to(device)
            with torch.inference_mode():
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.float16):
                    _, preds = model(h=sample, labels=label)

                for (_, fn), (_, meter) in zip(fns.items(), metrics.items()):
                    metric = fn(preds, label)
                    meter.update(metric.detach().item())

    return {k: meter.avg for k, meter in metrics.items()}

# --------------------------------------------------------

def get_loaders(args):
    modality_inds = defaultdict(lambda: [0,1])
    modality_inds["path_only"] = [0]
    modality_inds["rad_only"] = [1]

    index = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)
    inds = np.arange(len(index[args.label_col]))
    filter = (~np.isnan(index[args.label_col])) & (np.prod(index['combined_lengths'][:, modality_inds[args.emb_merge]], axis=1)).astype(bool)

    valid_inds = inds[filter]
    num_train = int(len(valid_inds) * args.train_split)
    num_valid = len(valid_inds) - num_train
    
    del index, inds, filter

    np.random.shuffle(valid_inds)
    train_inds, test_inds = valid_inds[:num_train], valid_inds[num_train:]

    dataset_args = {
        "data_dir": args.data_path,
        "max_instances": args.max_instances,
        "return_key": True,
        "label_column": args.label_col,
        "num_modalities": 1 if args.emb_merge in modality_inds else 2,
        "modality_inds": modality_inds[args.emb_merge]
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

    print(f"Found: {len(valid_inds)} valid samples split into "
        f"\n{num_train} train samples, {len(train_loader)} batches and "
        f"\n{num_valid} validation samples, {len(test_loader)} batches"
        # f"\nTrain: under 5 year: {np.sum(index[args.label_col][train_inds] == 0)}, over 5 year: {np.sum(index[args.label_col][train_inds] == 1)}"
        # f"\nTest: under 5 year: {np.sum(index[args.label_col][test_inds] == 0)}, over 5 year: {np.sum(index[args.label_col][test_inds] == 1)}"
    )

    return train_loader, test_loader, num_train

def get_full_loaders(args):
    modality_inds = defaultdict(lambda: [0,1])
    modality_inds["path_only"] = [0]
    modality_inds["rad_only"] = [1]

    index = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)
    inds = np.arange(len(index[args.label_col]))
    filter = (~np.isnan(index[args.censor_col])) & (~np.isnan(index[args.label_col])) & (np.prod(index['combined_lengths'][:, modality_inds[args.emb_merge]], axis=1)).astype(bool)

    valid_inds = inds[filter]
    num_train = int(len(valid_inds) * args.train_split)
    num_valid = len(valid_inds) - num_train
    
    del index, inds, filter

    np.random.shuffle(valid_inds)
    train_inds, test_inds = valid_inds[:num_train], valid_inds[num_train:]

    dataset_args = {
        "data_dir": args.data_path,
        "max_instances": args.max_instances,
        "return_key": False,
        "label_column": args.label_col,
        "num_modalities": 1 if args.emb_merge in modality_inds else 2,
        "modality_inds": modality_inds[args.emb_merge]
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

    print(f"Found: {len(valid_inds)} valid samples split into "
        f"\n{num_train} train samples, {len(train_loader)} batches and "
        f"\n{num_valid} validation samples, {len(test_loader)} batches"
        # f"\nTrain: under 5 year: {np.sum(index[args.label_col][train_inds] == 0)}, over 5 year: {np.sum(index[args.label_col][train_inds] == 1)}"
        # f"\nTest: under 5 year: {np.sum(index[args.label_col][test_inds] == 0)}, over 5 year: {np.sum(index[args.label_col][test_inds] == 1)}"
    )

    return train_loader, test_loader, num_train


# --------------------------------------------------------

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader, train_samples = get_loaders(args)
    iter_per_epoch = train_samples if args.per_sample else len(train_loader)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    merge_fn = {
        "sum": sum,
        "prod": lambda embs: torch.prod(torch.stack(embs), dim=0),
        "none": lambda embs: embs,
        "path_only": lambda embs: embs[0],
        "rad_only": lambda embs: embs[0],
    }

    modalities = defaultdict(lambda: "Path + Rad")
    modalities["path_only"] = "Path"
    modalities["rad_only"] = "Rad"
    

    losses = {
        "l1": F.l1_loss,
        "smooth_l1": F.smooth_l1_loss,
        "mse": F.mse_loss,
        "bce": F.binary_cross_entropy_with_logits,
    }

    model = EmbPred(loss_fn=losses[args.loss_fn])
    model = model.to(device)

    optimizer, scheduler = get_opt_and_sched(model, args, iter_per_epoch=iter_per_epoch)
    scaler = torch.amp.GradScaler(device, enabled=True)

    if args.disable_wandb: run = None
    else:
        config = {
            "Model": args.model,
            # "lr": args.lr,
            # **loader_args
            "Loss": args.loss_fn,
            "Seed": args.seed,
            "Batched Train": not args.per_sample,
            "LR Update": "Per batch",
            "Modalities": "Path & Rad",
            "Merge Func": args.emb_merge
        }


        run = wandb.init(
            entity="bumjin_joo-brown-university", 
            project=f"Panc MM 5yr Surv", 
            name=f"{args.model} Text - {modalities[args.emb_merge]} - {args.emb_merge} - {args.label_col}", 
            config=config
        )

    train_fn = train_one_epoch if args.per_sample else train_one_batched

    print(f"Start training for {args.epochs} epochs")
    pbar = trange(0, args.epochs, desc="Training Epochs", postfix={})
    for _ in pbar:
        train_stats = train_fn(model, train_loader, 
                               merge_fn[args.emb_merge],
                               optimizer, scheduler,
                               scaler, device, args)
        test_stats = test(model, test_loader, merge_fn[args.emb_merge], device, args=args)

        postfix = {**train_stats, **test_stats}
        if run is not None: run.log(postfix)
        pbar.set_postfix(postfix)
       

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    args.data_path = args.data_path.format(model=args.model)
    if args.debug:
        args.epochs = 1
        args.disable_wandb = True

    main(args)