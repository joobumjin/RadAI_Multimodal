from argparse import Namespace
from typing import Iterable
from collections import defaultdict

import wandb
from tqdm import trange
import numpy as np
from torch.nn import functional as F
from torch import optim

from data import *
from util import *

# --------------------------------------------------------

def get_bool_metrics(split: str):
    fns = {} 
    metrics = defaultdict(lambda: AverageMeter())

    return metrics, fns

def get_regression_metrics(split: str):
    fns = {"MSE": F.mse_loss, "L1": F.l1_loss}
    fns = {f"{split} {name}": fn for name, fn in fns.items()}

    metrics = defaultdict(lambda: AverageMeter())
    
    return metrics, fns

# --------------------------------------------------------

def train_one_epoch(model: torch.nn.Module,
                    train_loader: Iterable,
                    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler,
                    device: str,
                    args: Namespace):
    model.train(True)
    optimizer.zero_grad()

    metrics, fns = get_bool_metrics("Train")
    
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

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return {k: meter.avg for k, meter in metrics.items()}

def train_one_epoch_list(model: torch.nn.Module, 
                         train_loader: Iterable, 
                         optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, 
                         device: str, 
                         args: Namespace):
    model.train(True)
    optimizer.zero_grad()

    metrics, fns = get_bool_metrics("Train")
    
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

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return {k: meter.avg for k, meter in metrics.items()}

def test(model: torch.nn.Module, 
         data_loader: Iterable, 
         device: str, 
         args: Namespace,
         split: str = "Test"):
    model.eval()

    metrics, fns = get_bool_metrics(split)

    for batch in data_loader:
        for key in batch:
            batch[key] = batch[key].to(device)

        with torch.inference_mode():
            preds, loss = model(batch)
            metrics[f"{split} Loss"].update(loss.detach().item())
            for name, fn in fns.items():
                metric_val = fn(preds, batch["label"])
                metrics[name].update(metric_val.detach().item())

    return {k: meter.avg for k, meter in metrics.items()}

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
            "Mix Data": args.mix_data
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
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, device, args)
        if valid_loader is not None:
            valid_stats = test(model, valid_loader, device, args=args, split="Valid") 
        else:
            valid_stats = {}
        test_stats = test(model, test_loader, device, args=args, split="Test")

        c_ind_auc = calculate_c_indices_auc(model, train_loader, valid_loader, test_loader, device, surv_yr=args.survival_years)

        postfix = {**train_stats, **valid_stats, **test_stats, **c_ind_auc}

        if run is not None: 
            run.log(postfix)

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