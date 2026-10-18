from argparse import Namespace
from collections import defaultdict

import wandb
from tqdm import trange
import numpy as np
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader

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

def train_one_epoch(model: torch.nn.Module, device: torch.device, train_loader: DataLoader,
                    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler,):
    model.train(True)
    optimizer.zero_grad()

    metrics, fns = get_bool_metrics("Train")
    
    for batch in train_loader:
        for key in batch:
            batch[key] = batch[key].to(device)

        preds, loss = model(batch)
        metrics["Train Loss"].update(loss.detach().item())
        for name, fn in fns.items():
            metric_val = fn(preds, batch["label"])
            metrics[name].update(metric_val.detach().item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return {k: meter.avg for k, meter in metrics.items()}

def train_one_epoch_list(model: torch.nn.Module, device: torch.device, train_loader: DataLoader, 
                         optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, ):
    model.train(True)
    optimizer.zero_grad()

    metrics, fns = get_bool_metrics("Train")
    
    for batch in train_loader:
        for key in batch:
            batch[key] = batch[key].to(device)

        preds, loss = model(batch)

        metrics["Train Loss"].update(loss.detach().item())
        for name, fn in fns.items():
            metric_val = fn(preds, batch["label"])
            metrics[name].update(metric_val.detach().item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return {k: meter.avg for k, meter in metrics.items()}

def test(model: torch.nn.Module, device: torch.device, data_loader: DataLoader, split: str = "Test"):
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

def run_setup(args: Namespace, model_constructor: callable, 
              train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader, 
              run_name: Optional[str] = None, return_keys: list = [], ret_model: bool = False):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, device = model_constructor(args, train_loader.dataset._dtypes)

    optimizer, scheduler = get_opt_and_sched(model, args, iter_per_epoch=len(train_loader))

    early_stopper = EarlyStopper(args.patience, False) if args.early_stop else None
    stop_metric = "Test C-Index"

    if args.disable_wandb: run = None
    else:
        mods = []
        if args.clinical: mods.append("Clinical")
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
            config=vars(args)
        )

    ret_metrics = {key: [] for key in return_keys}

    pbar = trange(0, args.epochs, desc="Training Epochs", postfix={})
    for e in pbar:
        train_stats = train_one_epoch(model, device, train_loader, optimizer, scheduler)
        if valid_loader is not None:
            valid_stats = test(model, device, valid_loader, split="Valid") 
        else:
            valid_stats = {}
        test_stats = test(model, device, test_loader, split="Test")

        c_ind_auc = calculate_c_indices_auc(model, device, train_loader, valid_loader, test_loader, surv_yr=args.survival_years)

        postfix = {**train_stats, **valid_stats, **test_stats, **c_ind_auc}

        if run is not None: 
            run.log(postfix)

        pbar.set_postfix(postfix)

        for key in return_keys: ret_metrics[key].append(postfix[key])

        if early_stopper is not None:
            stop, _ = early_stopper.update(postfix[stop_metric])
            if stop:
                print("Early stopping triggered!")
                return ret_metrics

        if hasattr(args, "early") is not None and e == args.epochs + args.early: break

    if run is not None:
        run.finish()

    if ret_model: ret_metrics["model"] = model, device

    return ret_metrics