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

from torchmetrics import ROC, AUROC

from data import *
from model import create_mlp, EmbMIL, DenseFusion
from util import *

def get_args_parser():
    parser = argparse.ArgumentParser('Supervised Panc Training', add_help=False)

    parser.add_argument('--seed',               type=int,   default=0)
      
    parser.add_argument('--batch_size',         type=int,   default=32)
    parser.add_argument('--loss_fn',            type=str,   default="bce")
    parser.add_argument('--model',              type=str,   default="conch", choices=['conch', 'biomedclip'])
    parser.add_argument('--data_path',          type=str,   default="../{model}_path_rad_text_embs")
    parser.add_argument('--epochs',             type=int,   default=200)
    parser.add_argument('--device',                         default='cuda')
    parser.add_argument('--float16',            type=bool,  default=True)
    parser.add_argument('--early_stop',         type=bool,  default=True)
    parser.add_argument('--patience',           type=int,   default=5)

    parser.add_argument('--sparse',             action="store_true")
    parser.add_argument('--clinical',           action="store_true")
    parser.add_argument('--clinical_imputed',   action="store_true")
    parser.add_argument('--path_lang',          action="store_true")
    parser.add_argument('--rad_lang',           action="store_true")
    parser.add_argument('--path_img',           action="store_true")
    parser.add_argument('--emb_dim',            type=int,   default=64)
    
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
    # clin_enc = create_mlp(24, [128], args.emb_dim, act = nn.GELU(), dropout = 0.3, layer_norm = True)
    clin_enc = create_mlp(24, [128], args.emb_dim, act = nn.GELU(), dropout = 0.3, layer_norm = True)
    return clin_enc, False

def get_path_lang_encoder(args):
    # path_lang_enc = create_mlp(512, [128], args.emb_dim, act = nn.GELU(), dropout = 0.3, layer_norm = True)
    path_lang_enc = create_mlp(512, [128], args.emb_dim, act = nn.GELU(), dropout = 0.3, layer_norm = True)
    return path_lang_enc, True

def get_rad_lang_encoder(args):
    # rad_lang_enc = create_mlp(512, [128], args.emb_dim, act = nn.GELU(), dropout = 0.3, layer_norm = True)
    rad_lang_enc = create_mlp(512, [128], args.emb_dim, act = nn.GELU(), dropout = 0.3, layer_norm = True)
    return rad_lang_enc, True

def get_path_img_encoder(args):
    mil = EmbMIL(embed_dim=384, dropout=0.3, attn_dim = 256, proj_dim=args.emb_dim, loss_fn=None)
    return mil, True

def get_fusion_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    losses = {
        "l1": F.l1_loss,
        "smooth_l1": F.smooth_l1_loss,
        "mse": F.mse_loss,
        "bce": F.binary_cross_entropy_with_logits,
    }

    get_enc_fns = {
        "clinical": get_clinical_encoder, 
        "clinical_imputed": get_clinical_encoder, 
        "path_lang": get_path_lang_encoder, 
        "rad_lang": get_rad_lang_encoder, 
        "path_img": get_path_img_encoder, 
    }

    encs, casts = {}, {}
    arg_dict = vars(args)
    for mod, fn in get_enc_fns.items():
        if arg_dict.get(mod, False): 
            encs[mod], casts[mod] = fn(args)

    model = DenseFusion(encs, emb_dim=args.emb_dim, hidden_dims=[32], autocast=casts, loss_fn=losses[args.loss_fn], device=device)
    return model, device

# --------------------------------------------------------

def get_loaders(args):
    keys = ["slide_ids", "vital_status", "survival_months"]

    index = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)
    labels  = index['death_indicator_2yr'].astype(np.float32)
    inds    = np.arange(len(labels))
    bin_mods, extra_mods = [], []

    label_mask = ~np.isnan(index[args.label_col])
    exclusion_mask = ~index["excluded"]
    mask = label_mask & exclusion_mask
    if "indicator" in args.label_col: 
        for key in keys:
            mask = mask & (~np.isnan(index[key]))

    modality_mask = np.zeros_like(mask).astype(bool) if args.sparse else np.ones_like(mask).astype(bool)
    combine_op = lambda x, y: x | y if args.sparse else lambda x, y: x & y

    arg_dict = vars(args)
    for mod in ["clinical", "clinical_imputed"]:
        if arg_dict.get(mod, False): 
            modality_mask = combine_op(modality_mask, ~np.isnan(index[mod]).any(axis=1))
            extra_mods.append(mod)
    for mod, ind in zip(["path_lang", "rad_lang"], [0,1]):
        if arg_dict.get(mod, False): 
            modality_mask = combine_op(modality_mask, index['combined_lengths'][:, ind].astype(bool))
            bin_mods.append(mod)

    mask = mask & modality_mask

    valid_inds = inds[mask]
    num_train = int(len(valid_inds) * args.train_split)
    if not args.clinical_imputed: #standard random shuffle and select
        np.random.shuffle(valid_inds)
        
        train_inds, test_inds = valid_inds[:num_train], valid_inds[num_train:]
    else: #only test on real samples, but can train on imputed data
        orig = inds[~np.isnan(index['clinical']).any(axis=1) & mask]
        np.random.shuffle(orig)

        imputed = inds[np.isnan(index['clinical']).any(axis=1) & mask]

        num_test = len(inds) - num_train
        train_inds, test_inds = np.hstack((orig[num_test:], imputed)), orig[:num_test]


    dataset_args = {
        "data_dir": args.data_path,
        "return_key": True,
        "keys": ["slide_ids", "vital_status", "survival_months"],
        "label_column": args.label_col,
        "label_dtype": np.float32,
        "bin_modality_keys": bin_mods,
        "extra_modality_keys": extra_mods,
        "allow_sparse_samples": args.sparse
    }
    loader_args = {
        "batch_size": args.batch_size,
        "pin_memory": args.pin_mem,
        "num_workers": args.num_workers,
        "collate_fn": default_collate,
        "persistent_workers": args.num_workers > 0,
        "drop_last": False,
    }

    train_set = MemmapDatasetMultimodal(indices=train_inds, **dataset_args)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    test_set = MemmapDatasetMultimodal(indices=test_inds, **dataset_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    print(f"Found: {len(valid_inds)} valid samples split into "
        f"\n{len(train_set)} train samples, {len(train_loader)} batches and "
        f"\n{len(test_set)} validation samples, {len(test_loader)} batches"
        f"\nTrain: under 2 year: {np.sum(index[args.label_col][train_inds] == 1)}, over 2 year: {np.sum(index[args.label_col][train_inds] == 0)}"
        f"\nTest: under 2 year: {np.sum(index[args.label_col][test_inds] == 1)}, over 2 year: {np.sum(index[args.label_col][test_inds] == 0)}"
    )

    return train_loader, test_loader, num_train

# --------------------------------------------------------

def get_metrics(split: str, args):
    bool_var = "indicator" in args.label_col

    metrics = {f"{split} Loss": AverageMeter()}
    if split == "Train": metrics["lr"] = AverageMeter()
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

def test(model: torch.nn.Module, data_loader: Iterable, device: str, args=None):
    model.eval()

    metrics, fns, torchmetrics = get_metrics("Test", args)

    for batch in data_loader:
        for key in batch:
            batch[key] = batch[key].to(device)

        with torch.inference_mode():
            preds, loss = model(batch)
            metrics["Test Loss"].update(loss.detach().item())
            for name, fn in fns.items():
                metric_val = fn(preds, batch["label"])
                metrics[name].update(metric_val.detach().item())

            preds = torch.sigmoid(preds)
            for obj in torchmetrics.values():
                obj.update(preds.detach().squeeze(-1), batch["label"].detach().int().squeeze(-1))

    return {k: meter.avg for k, meter in metrics.items()}, torchmetrics


# --------------------------------------------------------

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader, _ = get_loaders(args)

    model, device = get_fusion_model(args)

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

        name = "+".join(mods) + f" - smaller, 64e - {args.label_col} - {args.model}"

        run = wandb.init(
            entity="bumjin_joo-brown-university", 
            project=f"Panc MM Cleanup Concat Fusion w Stage", 
            name=name,
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

        if early_stopper is not None:
            stop, best = early_stopper.update(postfix[stop_metric])
            if best:
                # save_model(model, args.save_path)
                pass
            elif stop:
                print("Early stopping triggered!")
                return
                
       

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    args.data_path = args.data_path.format(model=args.model)

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True

    main(args)