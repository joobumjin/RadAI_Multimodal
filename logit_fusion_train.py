import os
import argparse
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
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from torchmetrics import ROC, AUROC

from data import *
from model import LinearModel, EmbPred, EmbMIL, LogitFusion, NaiveSum, NaiveAvg, LearnedWeightSum
from util import *

def get_args_parser():
    parser = argparse.ArgumentParser('Supervised Panc Training', add_help=False)

    parser.add_argument('--seed',               type=int,   default=0)
      
    parser.add_argument('--batch_size',         type=int,   default=32)
    parser.add_argument('--loss_fn',            type=str,   default="bce")
    parser.add_argument('--data_path',          type=str,   default="../lab_clinical")
    parser.add_argument('--epochs',             type=int,   default=200)
    parser.add_argument('--device',                         default='cuda')
    parser.add_argument('--float16',            type=bool,  default=True)

    parser.add_argument('--clinical',           action="store_true")
    parser.add_argument('--path_lang',          action="store_true")
    parser.add_argument('--rad_lang',           action="store_true")
    parser.add_argument('--path_img',           action="store_true")
    parser.add_argument('--fusion',             type=str,   default="naive_sum", choices=["naive_sum", "naive_avg", "weighted_sum"])
    
    parser.add_argument('--prefetch_factor',    type=int,   default=2)
    parser.add_argument('--num_workers',        type=int,   default=1)
    parser.add_argument('--pin_mem',            type=bool,  default=True)
    parser.add_argument('--train_split',        type=float, default=.85)

    parser.add_argument('--label_col',          type=str,   default="death_indicator_2yr")
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

def get_clinical_encoder(args):
    hidden_dims = [64, 16]
    clin_enc = LinearModel(hidden_dims = hidden_dims, loss_fn=None)
    return clin_enc

def get_path_lang_encoder(args):
    path_lang_enc = EmbPred(loss_fn=None)
    return path_lang_enc

def get_rad_lang_encoder(args):
    rad_lang_enc = EmbPred(loss_fn=None)
    return rad_lang_enc

def get_path_img_encoder(args):
    mil = EmbMIL(loss_fn=None)
    return mil

def get_fusion_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fusers = {
        "naive_sum": NaiveSum,
        "naive_avg": NaiveAvg,
        "weight_sum": LearnedWeightSum
    }

    losses = {
        "l1": F.l1_loss,
        "smooth_l1": F.smooth_l1_loss,
        "mse": F.mse_loss,
        "bce": F.binary_cross_entropy_with_logits,
    }

    encs = {}
    arg_dict = vars(args)
    for mod, get_enc_fn in zip(["clinical", "path_lang", "rad_lang", "path_img"],
                               [get_clinical_encoder(args), get_path_lang_encoder(args), get_rad_lang_encoder(args), get_path_img_encoder(args)]):
        if arg_dict.get(mod, False):
            enc = get_enc_fn(args)
            enc = enc.to(device)
            encs[mod] = encs[mod].to(device)

    model = LogitFusion(encs, fusion_fn=fusers[args.fusion], loss_fn=losses[args.loss_fn])
    return model, device

def get_opt_and_sched(model, args, iter_per_epoch = None):
    optimizer = optim.AdamW({mod: enc.parameters() for mod, enc in model.encoders.items()}, lr=args.lr, betas=(0.9, 0.95))

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
    fns = {"Acc": lambda p, l: acc(torch.sigmoid(p) > 0.5, l)} if bool_var else {"MSE": F.mse_loss, "L1": F.l1_loss, "2yr Acc": lambda p, l: acc(p > 24, l > 24)}
    fns = {f"{split} {name}": fn for name, fn in fns.items()}
    test_metrics = {f"{name}": AverageMeter() for name in fns}
    metrics = {**metrics, **test_metrics}
    
    torchmetrics = {"ROC": ROC(task="binary"), "AUC": AUROC(task="binary")} if bool_var else {}
    torchmetrics = {f"{split} {name}": obj for name, obj in torchmetrics.items()}

    return metrics, fns, torchmetrics


def train_one_epoch(model: torch.nn.Module,
                    train_loader: Iterable,
                    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler,
                    device: str,
                    args=None):
    model.train(True)
    optimizer.zero_grad()

    metrics, fns, torchmetrics = get_metrics("Train", args)
    
    for (feats, labels, _) in train_loader:
        feats = feats.to(device)
        labels = labels.unsqueeze(-1)
        labels = labels.to(device)

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

def test(model: torch.nn.Module, data_loader: Iterable, device: str, args=None):
    model.eval()

    metrics, fns, torchmetrics = get_metrics("Test", args)

    for (feats, labels, _) in data_loader:
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

def get_loaders(args):
    keys = ["slide_ids", "vital_status", "survival_months"]

    index = np.load(f"{args.data_path}/lab_clin.npz", allow_pickle=True)
    feats   = index['clinical']
    labels  = index['death_indicator_2yr'].astype(np.float32)
    inds    = np.arange(len(labels))
    modalities = []
    mod_inds = []
    if args.path_lang: 
        modalities.append("path_lang")
        mod_inds.append(0)
    if args.rad_lang: 
        modalities.append("rad_lang")
        mod_inds.append(1)

    label_mask = ~np.isnan(index[args.label_col])
    exclusion_mask = ~index["excluded"]
    feat_mask = ~np.isnan(feats).any(axis=1)
    feat_mask = np.prod(index['combined_lengths'][:, mod_inds], axis=1).astype(bool)
    mask = label_mask & exclusion_mask & feat_mask
    if "indicator" in args.label_col: 
        for key in keys:
            mask = mask & (~np.isnan(index[key]))

    valid_inds = inds[mask]
    num_train = int(len(valid_inds) * args.train_split)
    
    np.random.shuffle(valid_inds)
    train_inds, test_inds = valid_inds[:num_train], valid_inds[num_train:]

    dataset_args = {
        "data_dir": args.data_path,
        "max_instances": args.max_instances,
        "return_key": True,
        "keys": ["slide_ids", "vital_status", "survival_months"],
        "label_column": "survival_months",
        "label_dtype": np.float32,
        "bin_modality_keys": ["path_text", "rad_text"],
        "extra_modality_keys": ['clinical']
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

    return train_loader, test_loader, num_train


# --------------------------------------------------------

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader, _ = get_loaders(args)

    model, device = get_fusion_model(args)

    optimizer, scheduler = get_opt_and_sched(model, args, iter_per_epoch=len(train_loader))

    if args.disable_wandb: run = None
    else:
        config = {
            "Loss": args.loss_fn,
            "Seed": args.seed,
        }

        run = wandb.init(
            entity="bumjin_joo-brown-university", 
            project=f"Panc MM 2yr Surv", 
            name=f"Clinical + Lab - {args.label_col}", 
            config=config
        )

    print(f"Start training for {args.epochs} epochs")
    pbar = trange(0, args.epochs, desc="Training Epochs", postfix={})
    for e in pbar:
        train_stats, train_tm = train_one_epoch(model, train_loader, optimizer, scheduler, device, args)
        test_stats, test_tm = test(model, test_loader, device, args=args)

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
            c_indices = calculate_c_indices(model, train_loader, test_loader, device)

        postfix = {**train_stats, **test_stats, **c_indices, **tm}
        if run is not None: run.log(postfix)
        pbar.set_postfix(postfix)
       

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    if args.debug:
        args.epochs = 1
        args.disable_wandb = True

    main(args)