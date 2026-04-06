import os
import argparse
from typing import Iterable, Optional
from collections import defaultdict

import wandb
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored
from torchmetrics import ROC, AUROC

from data import *
from model import EmbPred
from util import *

def get_args_parser():
    parser = argparse.ArgumentParser('Supervised ABMIL Panc Training', add_help=False)

    parser.add_argument('--seed',               type=int,   default=0)
    
    parser.add_argument('--model',              type=str,   default="conch", choices=['conch', 'biomedclip'])
    
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
    bool_var = "indicator" in args.label_col

    metrics = {f"{split} Loss": AverageMeter(), "lr": AverageMeter()}
    fns = {"Acc": lambda p, l: acc(torch.sigmoid(p) > 0.5, l)} if bool_var else {"MSE": F.mse_loss, "L1": F.l1_loss, "5yr Acc": lambda p, l: acc(p > 60, l > 60)}
    fns = {f"{split} {name}": fn for name, fn in fns.items()}
    test_metrics = {f"{name}": AverageMeter() for name in fns}
    metrics = {**metrics, **test_metrics}
    
    torchmetrics = {"ROC": ROC(task="binary"), "AUC": AUROC(task="binary")} if bool_var else {}
    torchmetrics = {f"{split} {name}": obj for name, obj in torchmetrics.items()}

    return metrics, fns, torchmetrics


def train_one_epoch(model: torch.nn.Module,
                    train_loader: Iterable, merge_fn,
                    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler,
                    device: str,
                    args=None):
    model.train(True)
    optimizer.zero_grad()

    metrics, fns, torchmetrics = get_metrics("Train", args)
    
    for batch in train_loader:
        feats, labels, _ = batch
        feats = merge_fn(feats).squeeze(dim=1)
        feats = feats.to(device)
        labels = labels.unsqueeze(-1)
        labels = labels.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.float16):
            loss, preds = model(h=feats, labels=labels)

            with torch.inference_mode():
                metrics["Train Loss"].update(loss.detach().item())
                metrics["lr"].update(optimizer.param_groups[0]["lr"])
                for name, fn in fns.items():
                    train_loss = fn(preds, labels)
                    metrics[name].update(train_loss.detach().item())

                preds = torch.sigmoid(preds)
                for obj in torchmetrics.values():
                    obj.update(preds.detach().squeeze(-1), labels.detach().int().squeeze(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return {k: meter.avg for k, meter in metrics.items()}, torchmetrics

def test(model: torch.nn.Module, data_loader: Iterable, merge_fn, device: str, args=None):
    model.eval()

    metrics, fns, torchmetrics = get_metrics("Test", args)

    for batch in data_loader:
        feats, labels, _ = batch
        feats = merge_fn(feats).squeeze(dim=1)
        feats = feats.to(device)
        labels = labels.unsqueeze(-1)
        labels = labels.to(device)
        with torch.inference_mode():
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.float16):
                preds = model.predict(h=feats)

            for (_, fn), (_, meter) in zip(fns.items(), metrics.items()):
                metric = fn(preds, labels)
                meter.update(metric.detach().item())

            preds = torch.sigmoid(preds)
            for obj in torchmetrics.values():
                obj.update(preds.detach().squeeze(-1), labels.detach().int().squeeze(-1))

    return {k: meter.avg for k, meter in metrics.items()}, torchmetrics

# --------------------------------------------------------

def calculate_c_indices(model: torch.nn.Module, train_loader, test_loader, merge_fn, device):
    model.eval()

    train_preds, train_deaths, train_times = [], [], []
    test_preds, test_deaths, test_times = [], [], []

    for batch in train_loader:
        feats, _, keys = batch
        train_deaths.append(~keys[1])
        train_times.append(keys[2])

        feats = merge_fn(feats).squeeze(dim=1)
        feats = feats.to(device)
        with torch.inference_mode():
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.float16):
                preds = model.predict(h=feats)

            preds = preds.detach().squeeze(-1).cpu().numpy()
            train_preds.append(preds)
    
    train = [np.concatenate(l) for l in [train_deaths, train_times, train_preds]]

    for batch in test_loader:
        feats, _, keys = batch
        test_deaths.append(~keys[1])
        test_times.append(keys[2])

        feats = merge_fn(feats).squeeze(dim=1)
        feats = feats.to(device)
        with torch.inference_mode():
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.float16):
                preds = model.predict(h=feats)

            preds = preds.detach().squeeze(-1).cpu().numpy()
            test_preds.append(preds)

    test = [np.concatenate(l) for l in [test_deaths, test_times, test_preds]]

    train_c, _, _, _, _ = concordance_index_censored(*train)
    test_c, _, _, _, _ = concordance_index_censored(*test)

    combined = [np.concatenate(arrs) for arrs in zip(train, test)]

    combined_c, _, _, _, _ = concordance_index_censored(*combined)

    return {"Train C-Index": train_c, "Test C-Index": test_c, "Combined C-Index": combined_c}

# --------------------------------------------------------

def get_loaders(args):
    modality_inds = defaultdict(lambda: [0,1])
    modality_inds["path_only"] = [0]
    modality_inds["rad_only"] = [1]
    keys = ["slide_ids", "vital_status", "survival_months"]

    index = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)
    inds = np.arange(len(index[args.label_col]))
    label_mask = ~np.isnan(index[args.label_col])
    exclusion_mask = ~index["excluded"]
    feat_mask = np.prod(index['combined_lengths'][:, modality_inds[args.emb_merge]], axis=1).astype(bool)
    mask = label_mask & exclusion_mask & feat_mask
    if "indicator" in args.label_col: 
        for key in keys:
            mask = mask & (~np.isnan(index[key]))

    valid_inds = inds[mask]
    num_train = int(len(valid_inds) * args.train_split)
    num_valid = len(valid_inds) - num_train
    
    del index, inds, mask

    np.random.shuffle(valid_inds)
    train_inds, test_inds = valid_inds[:num_train], valid_inds[num_train:]

    dataset_args = {
        "data_dir": args.data_path,
        "max_instances": args.max_instances,
        "return_key": True,
        "keys": keys,
        "label_column": args.label_col,
        "label_dtype": np.float32,
        "num_modalities": 1 if args.emb_merge in modality_inds else 2,
        "modality_inds": modality_inds[args.emb_merge]
    }
    loader_args = {
        "batch_size": args.batch_size,
        "pin_memory": args.pin_mem,
        "num_workers": args.num_workers,
        "collate_fn": collate_tensors,
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

    optimizer, scheduler = get_opt_and_sched(model, args, iter_per_epoch=len(train_loader))
    # scaler = torch.amp.GradScaler(device, enabled=True)

    if args.disable_wandb: run = None
    else:
        config = {
            "Model": args.model,
            # "lr": args.lr,
            # **loader_args
            "Loss": args.loss_fn,
            "Seed": args.seed,
            "Modalities": modalities[args.emb_merge],
            "Merge Func": args.emb_merge
        }

        run = wandb.init(
            entity="bumjin_joo-brown-university", 
            project=f"Panc MM 2yr Surv", 
            name=f"{args.model} Text - w Censored- {modalities[args.emb_merge]} - {args.emb_merge} - {args.label_col}", 
            config=config
        )

    print(f"Start training for {args.epochs} epochs")
    pbar = trange(0, args.epochs, desc="Training Epochs", postfix={})
    for e in pbar:
        train_stats, train_tm = train_one_epoch(model, train_loader, 
                               merge_fn[args.emb_merge],
                               optimizer, scheduler,
                               device, args)
        test_stats, test_tm = test(model, test_loader, merge_fn[args.emb_merge], device, args=args)

        tm = {}
        if len(train_tm) > 0:
            if e == args.epochs - 1:
                if run is not None: 
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.suptitle('Train and Test ROC Curves')
                    train_tm["Train ROC"].plot(ax=ax1)
                    test_tm["Test ROC"].plot(ax=ax2)
                    run.log({"ROC": fig})
            del train_tm["Train ROC"], test_tm["Test ROC"]

            tm = {**train_tm, **test_tm}
            tm = {name: obj.compute() for name, obj in tm.items()}

        c_indices = {}
        if "indicator" in args.label_col:
            c_indices = calculate_c_indices(model, train_loader, test_loader, merge_fn[args.emb_merge], device)

        postfix = {**train_stats, **test_stats, **c_indices, **tm}
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