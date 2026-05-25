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
from model import *
from util import *
from run import *

def get_args_parser():
    parser = argparse.ArgumentParser('Supervised Panc Training', add_help=False)

    parser.add_argument('--seed',               type=int,   default=0)
      
    parser.add_argument('--batch_size',         type=int,   default=16)
    parser.add_argument('--loss_fn',            type=str,   default="bce")
    parser.add_argument('--model',              type=str,   default="conch", choices=['conch', 'biomedclip', 'gemma', 'qwen'])
    parser.add_argument('--data_path',          type=str,   default="../updated_multimodal_bins")
    parser.add_argument('--test_path',          type=str,   default="../multimodal_bins_rw")
    parser.add_argument('--epochs',             type=int,   default=200)
    parser.add_argument('--device',                         default='cuda')
    parser.add_argument('--float16',            type=bool,  default=True)
    parser.add_argument('--early_stop',         type=bool,  default=True)
    parser.add_argument('--patience',           type=int,   default=5)

    parser.add_argument('--sparse',             action="store_true")
    parser.add_argument('--fusion',             type=str,   default="naive_sum", choices=["naive_sum", "naive_avg", "weighted_sum"])
    parser.add_argument('--clinical',           action="store_true")
    parser.add_argument('--path_lang',          action="store_true")
    parser.add_argument('--rad_lang',           action="store_true")
    parser.add_argument('--path_img',           action="store_true")
    parser.add_argument('--enc_dim',            type=int,   default=512)
    parser.add_argument('--emb_dim',            type=int,   default=256)
    
    parser.add_argument('--prefetch_factor',    type=int,   default=2)
    parser.add_argument('--num_workers',        type=int,   default=1)
    parser.add_argument('--pin_mem',            type=bool,  default=True)
    parser.add_argument('--train_split',        type=float, default=.85)

    parser.add_argument('--label_col',          type=str,   default="survival_days")
    parser.add_argument('--survival_years',     type=int,   default=2)
    parser.add_argument('--censor_col',         type=str,   default="right_censor")

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
    parser.add_argument('--wb_proj',            type=str,   default="Panc MultiLoss")
    parser.add_argument('--debug',              action="store_true")
    return parser

# --------------------------------------------------------

def get_inds(args):
    keys = ["slide_ids", "survival_days", "survival_right_censor"]

    index   = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)
    inds    = np.arange(len(index['survival_days']))

    label_mask = index[f"{args.label_col}_mask"]
    exclusion_mask = ~index["excluded"]
    mask = label_mask & exclusion_mask
    for key in keys:
        mask = mask & (~np.isnan(index[key]))

    modality_mask = np.zeros_like(mask).astype(bool) if args.sparse else np.ones_like(mask).astype(bool)
    combine_op = (lambda x, y: x | y) if args.sparse else (lambda x, y: x & y)

    arg_dict = vars(args)
    for mod in ["clinical", "path_lang", "rad_lang", "path_img"]:
        if arg_dict.get(mod, False): 
            modality_mask = combine_op(modality_mask, (index[f'{mod}_mask']))

    mask = mask & modality_mask

    valid_inds = inds[mask]
    num_train = int(len(valid_inds) * args.train_split)
    np.random.shuffle(valid_inds)
    
    train_inds, validation_inds = valid_inds[:num_train], valid_inds[num_train:]

    return train_inds, validation_inds

# --------------------------------------------------------

def get_clinical_encoder(args):
    # clin_enc = create_mlp(24, [128], args.emb_dim, act = nn.GELU(), dropout = 0.3, layer_norm = True)
    clin_enc = create_mlp(24, [128], args.emb_dim, act = nn.GELU(), dropout = 0.4, end_with_fc=False, end_with_dropout=True, layer_norm = True, end_with_norm=True)
    return clin_enc, False

def get_path_lang_encoder(args):
    # path_lang_enc = create_mlp(args.enc_dim, [128], args.emb_dim, act = nn.GELU(), dropout = 0.4, layer_norm = True)
    path_lang_enc = create_mlp(args.enc_dim, [128], args.emb_dim, act = nn.GELU(), dropout = 0.4, end_with_fc=False, end_with_dropout=True, layer_norm = True, end_with_norm=True)
    return path_lang_enc, True

def get_rad_lang_encoder(args):
    # rad_lang_enc = create_mlp(args.enc_dim, [128], args.emb_dim, act = nn.GELU(), dropout = 0.4, layer_norm = True)
    rad_lang_enc = create_mlp(args.enc_dim, [128], args.emb_dim, act = nn.GELU(), dropout = 0.4, end_with_fc=False, end_with_dropout=True, layer_norm = True, end_with_norm=True)
    return rad_lang_enc, True

def get_path_img_encoder(args):
    mil = EmbMIL(embed_dim=384, dropout=0.3, attn_dim = 256, proj_dim=args.emb_dim, loss_fn=None)
    return mil, True

def get_surv_day_decoder(args):
    dec = create_mlp(args.emb_dim, [128], 1, act = nn.GELU(), dropout = 0.3, layer_norm = True, end_with_norm=True)
    return dec, False

def get_surv_indic_decoder(args):
    dec = create_mlp(args.emb_dim, [128], 1, act = nn.GELU(), dropout = 0.3, layer_norm = True, end_with_norm=True)
    return dec, False

def get_recur_day_decoder(args):
    dec = create_mlp(args.emb_dim, [128], 1, act = nn.GELU(), dropout = 0.3, layer_norm = True, end_with_norm=True)
    return dec, False

def get_clin_decoder(args):
    dec = create_mlp(args.emb_dim, [128], 24, act = nn.GELU(), dropout = 0.3, layer_norm = True, end_with_norm=True)
    return dec, False

def get_path_lang_decoder(args):
    dec = create_mlp(args.emb_dim, [128], args.enc_dim, act = nn.GELU(), dropout = 0.3, layer_norm = True, end_with_norm=True)
    return dec, True

def get_rad_lang_decoder(args):
    dec = create_mlp(args.emb_dim, [128], args.enc_dim, act = nn.GELU(), dropout = 0.3, layer_norm = True, end_with_norm=True)
    return dec, True

def get_encoders(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    get_enc_fns = {
        "clinical": get_clinical_encoder, 
        "path_lang": get_path_lang_encoder, 
        "rad_lang": get_rad_lang_encoder, 
        "path_img": get_path_img_encoder, 
    }

    encs, casts = {}, {}
    arg_dict = vars(args)
    for mod, fn in get_enc_fns.items():
        if arg_dict.get(mod, False): 
            encs[mod], casts[mod] = fn(args)
            encs[mod] = encs[mod].to(device)

    return encs, casts, device

def get_decoders(args, bool_targets: list[str], regr_targets: list[str], recon_targets: list[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #add decoders
    decoders = {
        "survival_days": get_surv_day_decoder, #-> single regression value
        "survival_2yr": get_surv_indic_decoder, #-> single probability
        "recur_free_days": get_recur_day_decoder, #-> single regression value
        "clinical": get_clin_decoder, #-> 24dim
        "path_lang": get_path_lang_decoder, #-> 512 dim
        "rad_lang": get_rad_lang_decoder #-> 512 dim
    }

    decs, casts = {}, {}
    for target, fn in decoders.items():
        if target in bool_targets + regr_targets + recon_targets:
            decs[target], casts[target] = fn(args)
            decs[target] = decs[target].to(device)

    return decs, casts, device

class MultiLossFn(nn.Module):
    def __init__(self, bool_targets: list[str], regr_targets: list[str], recon_targets: list[str], weights: dict[str, int], autocast: dict[str, bool], device):
        super(MultiLossFn, self).__init__()

        self.bool_targets = bool_targets
        self.bool_fn = F.binary_cross_entropy_with_logits

        self.regr_targets = regr_targets
        self.regr_fn = F.smooth_l1_loss

        self.recon_targets = recon_targets
        self.recon_fn = F.mse_loss

        self.weights = weights
        self.autocast = autocast
        self.device = device
    
    def forward(self, predictions, targets):
        total_loss, loss = 0.0, {}

        for target_split, loss_fn in zip([self.bool_targets, self.regr_targets, self.recon_targets], [self.bool_fn, self.regr_fn, self.recon_fn]):
            for target in target_split:
                if len(targets[f"{target}_mask"].shape) == 1: targets[f"{target}_mask"] = targets[f"{target}_mask"].view((-1, 1))
                
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[target]):
                    l = loss_fn(predictions[target], targets[target], reduction="none").float()

                unmasked_samples = targets[f"{target}_mask"].sum()
                loss_value = (l * targets[f"{target}_mask"]).sum() / unmasked_samples if unmasked_samples > 0 else torch.tensor(0.0, dtype=torch.float32, device=l.device)

                total_loss += self.weights[target] * loss_value
                loss[target] = loss_value.detach().cpu().numpy()

        loss["total_loss"] = total_loss

        return loss

# --------------------------------------------------------

def get_metrics(split: str, bool_targets: list[str], regr_targets: list[str], recon_targets: list[str]):
    metrics = {f"{split} Loss": AverageMeter()}
    if split == "Train": metrics["lr"] = AverageMeter()

    fns, torchmetrics = {}, {}

    #maybe split into bool and regression metric dicts
    for target in bool_targets:
        fns[target] = {
            f"{target} {split} Acc": lambda p, l: acc(torch.sigmoid(p) > 0.5, l)
        }
        torchmetrics[target] = {
            f"{target} {split} ROC": ROC(task="binary"),
            f"{target} {split} AUC": AUROC(task="binary")
        }
        
        metrics[f"{target} {split} Acc"] = AverageMeter()
        metrics[f"{target} {split} Loss"] = AverageMeter()

    for target in regr_targets + recon_targets:
        # fns[target] = {
        #     f"{target} {split} MSE": F.mse_loss,
        #     f"{target} {split} L1": F.l1_loss
        # }
        # metrics[f"{target} {split} MSE"] = AverageMeter()
        # metrics[f"{target} {split} L1"] = AverageMeter()
        metrics[f"{target} {split} Loss"] = AverageMeter()

    return metrics, fns, torchmetrics

def train_one_epoch(model: torch.nn.Module,
                    train_loader: Iterable,
                    bool_targets: list[str], regr_targets: list[str], recon_targets: list[str],
                    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler,
                    device: str,
                    args: Namespace):
    model.train(True)
    optimizer.zero_grad()

    metrics, fns, torchmetrics = get_metrics("Train", bool_targets, regr_targets, recon_targets)
    
    for batch in train_loader:
        for key in batch:
            batch[key] = batch[key].to(device)

        preds, loss = model(batch)

        #need to convert preds to a dict
        with torch.inference_mode():
            metrics["Train Loss"].update(loss["total_loss"].detach().cpu().item())
            metrics["lr"].update(optimizer.param_groups[0]["lr"])
            for target in bool_targets + regr_targets + recon_targets:
                metrics[f"{target} Train Loss"].update(loss[target].item())
            for target in bool_targets:
                for name, fn in fns[target].items():
                    metric_val = fn(preds[target], batch[target])
                    metrics[name].update(metric_val.detach().cpu().item())
            
                surv_pred = torch.sigmoid(preds[target])
                for obj in torchmetrics[target].values():
                    obj.update(surv_pred.detach().squeeze(-1), batch[target].detach().int().squeeze(-1))

        loss["total_loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return {k: meter.avg for k, meter in metrics.items()}, torchmetrics

def test(model: torch.nn.Module, 
         data_loader: Iterable, 
         bool_targets: list[str], regr_targets: list[str], recon_targets: list[str],
         device: str, args: Namespace, split: str = "Test"):
    model.eval()

    metrics, fns, torchmetrics = get_metrics(split, bool_targets, regr_targets, recon_targets)

    for batch in data_loader:
        for key in batch:
            batch[key] = batch[key].to(device)

        with torch.inference_mode():
            preds, loss = model(batch)
            metrics[f"{split} Loss"].update(loss["total_loss"].detach().cpu().item())
            for target in bool_targets + regr_targets + recon_targets:
                metrics[f"{target} {split} Loss"].update(loss[target].item())
            
            for target in bool_targets:
                for name, fn in fns[target].items():
                    metric_val = fn(preds[target], batch[target])
                    metrics[name].update(metric_val.detach().cpu().item())

                surv_pred = torch.sigmoid(preds[target])
                for obj in torchmetrics[target].values():
                    obj.update(surv_pred.detach().squeeze(-1), batch[target].detach().int().squeeze(-1))

    return {k: meter.avg for k, meter in metrics.items()}, torchmetrics

def compile_preds(model, loader, device):
    split_preds, split_deaths, split_times = [], [], []
    for batch in loader:
        surviving = batch["survival_right_censor"].numpy().squeeze(-1).astype(bool)
        times = batch["survival_days"].numpy().squeeze(-1)
        split_deaths.append(~surviving)
        split_times.append(times)

        for key in batch: batch[key] = batch[key].to(device)
        with torch.inference_mode():
            preds = model.predict(batch)["survival_2yr"]

            preds = preds.detach().squeeze(-1).cpu().numpy()
            split_preds.append(preds)
    
    return [np.concatenate(l) for l in [split_deaths, split_times, split_preds]]

def run_setup(args, model, device, bool_targets, regr_targets, recon_targets, train_loader, valid_loader, test_loader, run = None):
    optimizer, scheduler = get_opt_and_sched(model, args, iter_per_epoch=len(train_loader))

    early_stopper = EarlyStopper(args.patience, False) if args.early_stop else None
    stop_metric = "Test C-Index"

    print(f"Start training for {args.epochs} epochs")
    pbar = trange(0, args.epochs, desc="Training Epochs", postfix={})
    for e in pbar:
        train_stats, train_tm = train_one_epoch(model, train_loader, bool_targets, regr_targets, recon_targets, optimizer, scheduler, device, args)
        valid_stats, valid_tm = test(model, valid_loader, bool_targets, regr_targets, recon_targets, device, args=args, split="Valid")
        test_stats, test_tm = test(model, test_loader, bool_targets, regr_targets, recon_targets, device, args=args, split="Test")

        tm = {}
        if len(train_tm) > 0:
            if e == args.epochs - 1:
                for target in bool_targets:
                    if run is not None: 
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                        fig.suptitle('ROC Performance Curves')
                        train_tm[target][f"{target} Train ROC"].plot(ax=ax1)
                        valid_tm[target][f"{target} Valid ROC"].plot(ax=ax2)
                        test_tm[target][f"{target} Test ROC"].plot(ax=ax3)
                        run.log({f"{target} ROC": fig})
                del train_tm[target][f"{target} Train ROC"], valid_tm[target][f"{target} Valid ROC"], test_tm[target][f"{target} Test ROC"]

            tms = {**train_tm, **valid_tm, **test_tm}
            for t_d in tms.values():
                for name, obj in t_d.items():
                    tm[name] = obj.compute() 

        #need to adjust this
        c_indices = calculate_c_indices(model, train_loader, valid_loader, test_loader, device, compile=compile_preds)

        postfix = {**train_stats, **valid_stats, **test_stats, **c_indices, **tm}
        if run is not None: run.log(postfix)
        pbar.set_postfix(postfix)

        if early_stopper is not None:
            stop, best = early_stopper.update(postfix[stop_metric])
            if best:
                pass
            elif stop:
                print("Early stopping triggered!")
                return


# --------------------------------------------------------

def main(args):
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

        mods = [mod for mod, used 
                        in zip(["Clinical", "Path Lang", "Rad Lang", "Path Img"], 
                               [args.clinical, args.path_lang, args.rad_lang, args.path_img]) 
                        if used]

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

    bool_targets, regr_targets, recon_targets = ["survival_2yr"], ["survival_days", "recur_free_days"], ["clinical", "path_lang", "rad_lang"]

    encoders, enc_casts, device = get_encoders(args)
    decoders, dec_casts, device = get_decoders(args, bool_targets, regr_targets, recon_targets)
    casts = {**enc_casts, **dec_casts}

    arg_dict = vars(args)
    mods = [mod for mod in ["path_lang", "rad_lang", "clinical"] if arg_dict.get(mod, False)]

    train_inds, val_inds = get_inds(args)

    index = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)
    inds = np.arange(len(index['survival_days']))[~index["excluded"]]

    expanded_train = np.setdiff1d(inds, val_inds)

    #pretrain modality specifics
    for mod in mods:
        kw_mod = {"bin_mods": [mod]} if mod != "clinical" else {"extra_mods": mod}
        train_loader, valid_loader, test_loader = get_input_loader(args, train_inds = expanded_train, validation_inds=val_inds, **kw_mod)

        #train encoders[mod] and decoders[mod]
        model = nn.Sequential(encoders[mod], decoders[mod])
        optimizer, scheduler = get_opt_and_sched(model, args, iter_per_epoch=len(train_loader))

        pbar = trange(0, args.epochs, desc="Pretrain Reconstruction", postfix={})
        for _ in pbar:
            train_stats, _ = train_one_epoch(model, train_loader, [], [], [mod], optimizer, scheduler, device, args)
            valid_stats, _ = test(model, valid_loader, [], [], [mod], device, args=args, split="Valid")
            test_stats, _ = test(model, test_loader, [], [], [mod], device, args=args, split="Test")

            postfix = {**train_stats, **valid_stats, **test_stats}
            if run is not None: run.log(postfix)
            pbar.set_postfix(postfix)

    #pretrain survival regr, recurrence regr
    #need to disable bool decoder and recon decoders
    multimodal, _ = DenseFusionMulti(encoders, emb_dim=args.emb_dim, hidden_dims=[32], decoders=decoders, autocast=casts, loss_fn=None, device=device, targets=regr_targets)
    loss_weights = {
        "survival_days": .05, 
        "recur_free_days": .03, 
    }
    loss_fn = MultiLossFn([], regr_targets, [], weights=loss_weights, autocast=casts, device=device)
    multimodal.loss_fn = loss_fn
    optimizer, scheduler = get_opt_and_sched(model, args, iter_per_epoch=len(train_loader))

    train_loader, valid_loader, test_loader = get_key_loaders(args, train_inds = expanded_train, validation_inds=val_inds, keys=regr_targets + [f"{t}_mask" for t in regr_targets])

    pbar = trange(0, args.epochs, desc="Pretrain Regression", postfix={})
    for _ in pbar:
        train_stats, _ = train_one_epoch(model, train_loader, [], regr_targets, [], optimizer, scheduler, device, args)
        valid_stats, _ = test(model, valid_loader, [], regr_targets, [], device, args=args, split="Valid")
        test_stats, _ = test(model, test_loader, [], regr_targets, [], device, args=args, split="Test")

        postfix = {**train_stats, **valid_stats, **test_stats}
        if run is not None: run.log(postfix)
        pbar.set_postfix(postfix)


    loss_weights = {
        "survival_2yr": 10, 
        "survival_days": .05, 
        "recur_free_days": .03, 
    }
    loss_fn = MultiLossFn([], regr_targets, bool_targets, weights=loss_weights, autocast=casts, device=device)
    multimodal.loss_fn = loss_fn
    multimodal.targets = regr_targets + bool_targets

    train_loader, valid_loader, test_loader = get_loaders(args, train_inds, val_inds,
                                                          keys = ["survival_days", "survival_days_mask", "survival_right_censor", "recur_free_days", "recur_free_days_mask"],
                                                          label_key="survival_2yr")
    run_setup(args, multimodal, device, bool_targets, regr_targets, [], train_loader, valid_loader, test_loader, run=run)

    if run is not None:
        run.finish()

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    main(args)