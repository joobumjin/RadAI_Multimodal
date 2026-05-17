import os
import argparse
from argparse import Namespace
from typing import Iterable, Optional
from collections import defaultdict

import wandb
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from torch import optim

from torchmetrics import ROC, AUROC

from data import *
from model import create_mlp, EmbMIL, DenseFusion
from util import *

# --------------------------------------------------------

def get_bool_metrics(split: str):
    metrics = {f"{split} Loss": AverageMeter()}
    if split == "Train": metrics["lr"] = AverageMeter()
    fns = {"Acc": lambda p, l: acc(torch.sigmoid(p) > 0.5, l)} 
    fns = {f"{split} {name}": fn for name, fn in fns.items()}
    test_metrics = {f"{name}": AverageMeter() for name in fns}
    metrics = {**metrics, **test_metrics}
    
    torchmetrics = {"ROC": ROC(task="binary"), "AUC": AUROC(task="binary")}
    torchmetrics = {f"{split} {name}": obj for name, obj in torchmetrics.items()}

    return metrics, fns, torchmetrics

def get_regression_metrics(split: str):
    metrics = {f"{split} Loss": AverageMeter()}
    if split == "Train": metrics["lr"] = AverageMeter()
    fns = {"MSE": F.mse_loss, "L1": F.l1_loss}
    fns = {f"{split} {name}": fn for name, fn in fns.items()}
    test_metrics = {f"{name}": AverageMeter() for name in fns}
    metrics = {**metrics, **test_metrics}
    
    torchmetrics = {}

    return metrics, fns, torchmetrics

# --------------------------------------------------------

def train_one_epoch(model: torch.nn.Module,
                    train_loader: Iterable,
                    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler,
                    device: str,
                    args: Namespace):
    model.train(True)
    optimizer.zero_grad()

    metrics, fns, torchmetrics = get_bool_metrics("Train")
    
    for batch in train_loader:
        for key in batch:
            batch[key] = batch[key].to(device)

        preds, loss = model(batch)

        with torch.inference_mode():
            metrics["Train Loss"].update(loss.detach().item())
            metrics["lr"].update(optimizer.param_groups[0]["lr"])
            for name, fn in fns.items():
                metric_val = fn(preds, batch["label"])
                metrics[name].update(metric_val.detach().item())

            preds = torch.sigmoid(preds)
            for obj in torchmetrics.values():
                obj.update(preds.detach().squeeze(-1), batch["label"].detach().int().squeeze(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return {k: meter.avg for k, meter in metrics.items()}, torchmetrics

def train_one_epoch_list(model: torch.nn.Module, 
                         train_loader: Iterable, 
                         optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, 
                         device: str, 
                         args: Namespace):
    model.train(True)
    optimizer.zero_grad()

    metrics, fns, torchmetrics = get_bool_metrics("Train", args)
    
    for batch in train_loader:
        for key in batch:
            batch[key] = batch[key].to(device)

        preds, loss = model(batch)

        with torch.inference_mode():
            metrics["Train Loss"].update(loss.detach().item())
            metrics["lr"].update(optimizer.param_groups[0]["lr"])
            for name, fn in fns.items():
                metric_val = fn(preds, batch["label"])
                metrics[name].update(metric_val.detach().item())

            preds = torch.sigmoid(preds)
            for obj in torchmetrics.values():
                obj.update(preds.detach().squeeze(-1), batch["label"].detach().int().squeeze(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return {k: meter.avg for k, meter in metrics.items()}, torchmetrics

def test(model: torch.nn.Module, 
         data_loader: Iterable, 
         device: str, 
         args: Namespace,
         split: str = "Test"):
    model.eval()

    metrics, fns, torchmetrics = get_bool_metrics(split, args)

    for batch in data_loader:
        for key in batch:
            batch[key] = batch[key].to(device)

        with torch.inference_mode():
            preds, loss = model(batch)
            metrics[f"{split} Loss"].update(loss.detach().item())
            for name, fn in fns.items():
                metric_val = fn(preds, batch["label"])
                metrics[name].update(metric_val.detach().item())

            preds = torch.sigmoid(preds)
            for obj in torchmetrics.values():
                obj.update(preds.detach().squeeze(-1), batch["label"].detach().int().squeeze(-1))

    return {k: meter.avg for k, meter in metrics.items()}, torchmetrics

# --------------------------------------------------------

def run_setup(args, model_constructor, train_loader, valid_loader, test_loader, run_name = None):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, device = model_constructor(args)

    optimizer, scheduler = get_opt_and_sched(model, args, iter_per_epoch=len(train_loader))

    early_stopper = EarlyStopper(args.patience, False) if args.early_stop else None
    stop_metric = "Test C-Index"

    if args.disable_wandb: run = None
    else:
        config = {
            "Loss": args.loss_fn,
            "Seed": args.seed,
            "Clinical": args.clinical,
            "Clinical Imputed": args.clinical_imputed,
            "Path Lang": args.path_lang,
            "Rad Lang": args.rad_lang,
            "Path Img": args.path_img,
            "Fusion": "Dense",
            "Model": args.model,
            "MLP Norm": "Layer Norm",
        }

        mods = []
        if args.clinical: mods.append("Clinical")
        if args.clinical_imputed: mods.append("ClinImp")
        if args.path_lang: mods.append("Path Lang")
        if args.rad_lang: mods.append("Rad Lang")
        if args.path_img: mods.append("Path Img")

        if run_name is None:
            run_name =  " - ".join([f"{args.label_col}", f"{args.model}"]) + "+".join(mods)
        else:
            run_name += "+".join(mods)

        run = wandb.init(
            entity="bumjin_joo-brown-university", 
            project=args.wb_proj, 
            name=run_name,
            config=config
        )

    print(f"Start training for {args.epochs} epochs")
    pbar = trange(0, args.epochs, desc="Training Epochs", postfix={})
    for e in pbar:
        train_stats, train_tm = train_one_epoch(model, train_loader, optimizer, scheduler, device, args)
        valid_stats, valid_tm = test(model, valid_loader, device, args=args, split="Valid")
        test_stats, test_tm = test(model, test_loader, device, args=args, split="Test")

        tm = {}
        if len(train_tm) > 0:
            if e == args.epochs - 1:
                if run is not None: 
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    fig.suptitle('ROC Performance Curves')
                    train_tm["Train ROC"].plot(ax=ax1)
                    valid_tm["Valid ROC"].plot(ax=ax2)
                    test_tm["Test ROC"].plot(ax=ax3)
                    run.log({"ROC": fig})
            del train_tm["Train ROC"], valid_tm["Valid ROC"], test_tm["Test ROC"]

            tm = {**train_tm, **valid_tm}
            tm = {name: obj.compute() for name, obj in tm.items()}

        c_indices = calculate_c_indices(model, train_loader, valid_loader, test_loader, device)

        postfix = {**train_stats, **valid_stats, **test_stats, **c_indices, **tm}
        if run is not None: run.log(postfix)
        pbar.set_postfix(postfix)

        if early_stopper is not None:
            stop, best = early_stopper.update(postfix[stop_metric])
            if best:
                # save_model(model, args.save_path)
                pass
            elif stop:
                print("Early stopping triggered!")
                return
            
    if run is not None:
        run.finish()