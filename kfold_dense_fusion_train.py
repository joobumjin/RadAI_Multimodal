import os
import argparse
from argparse import Namespace
from typing import Iterable, Optional
from collections import defaultdict

import wandb
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim

from torchmetrics import ROC, AUROC

from data import *
from model import create_mlp, EmbMIL, DenseFusion
from util import *
from run import *
from dense_fusion_train import get_dense_fusion_model
from logit_fusion_train import get_logit_fusion_model


def get_args_parser():
    parser = argparse.ArgumentParser('Supervised Panc Training', add_help=False)

    parser.add_argument('--seed',               type=int,   default=0)
      
    parser.add_argument('--batch_size',         type=int,   default=16)
    parser.add_argument('--loss_fn',            type=str,   default="bce")
    parser.add_argument('--model',              type=str,   default="conch", choices=['conch', 'biomedclip'])
    parser.add_argument('--data_path',          type=str,   default="../updated_multimodal_bins")
    parser.add_argument('--test_path',          type=str,   default="../multimodal_bins_rw")
    parser.add_argument('--epochs',             type=int,   default=200)
    parser.add_argument('--device',                         default='cuda')
    parser.add_argument('--float16',            type=bool,  default=True)
    parser.add_argument('--early_stop',         type=bool,  default=True)
    parser.add_argument('--patience',           type=int,   default=5)
    parser.add_argument('--folds',              type=int,   default=5)

    parser.add_argument('--sparse',             action="store_true")
    parser.add_argument('--clinical',           action="store_true")
    parser.add_argument('--clinical_imputed',   action="store_true")
    parser.add_argument('--path_lang',          action="store_true")
    parser.add_argument('--rad_lang',           action="store_true")
    parser.add_argument('--path_img',           action="store_true")
    parser.add_argument('--emb_dim',            type=int,   default=64)
    parser.add_argument('--fusion_tech',        type=str,   default="emb_catdense", choices=["emb_catdense", "logit_naive_sum", "logit_naive_avg", "logit_weighted_sum"])
    
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
    parser.add_argument('--wb_proj',            type=str,   default="Panc MM Fusion Cross Validation External Test")
    parser.add_argument('--debug',              action="store_true")
    return parser


# --------------------------------------------------------

def get_inds(args):
    keys = ["slide_ids", "survival_days", "survival_right_censor"]

    index   = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)
    labels  = index['survival_days'].astype(np.float32)
    inds    = np.arange(len(labels))
    bin_mods, extra_mods = [], []

    label_mask = ~np.isnan(index[args.label_col])
    exclusion_mask = ~index["excluded"]
    mask = label_mask & exclusion_mask
    for key in keys:
        mask = mask & (~np.isnan(index[key]))

    modality_mask = np.zeros_like(mask).astype(bool) if args.sparse else np.ones_like(mask).astype(bool)
    combine_op = lambda x, y: x | y if args.sparse else x & y

    arg_dict = vars(args)
    for mod in ["clinical", "clinical_imputed"]:
        if arg_dict.get(mod, False): 
            modality_mask = combine_op(modality_mask, ~index[f'{mod}_mask'])
            extra_mods.append(mod)
    for mod in ["path_lang", "rad_lang", "path_img"]:
        if arg_dict.get(mod, False): 
            modality_mask = combine_op(modality_mask, ~(index[f'{mod}_mask']))
            bin_mods.append(mod)

    mask = mask & modality_mask

    valid_inds = inds[mask]
    np.random.shuffle(valid_inds)
    kf = KFold(n_splits=args.folds)
    
    # kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    return kf.split(X=valid_inds), valid_inds

# --------------------------------------------------------

def main(args):
    f_type, _, f_spec = args.fusion_tech.partition("_")
    match f_type:
        case "emb":
            model_constructor = get_dense_fusion_model
        case "logit":
            model_constructor = get_logit_fusion_model
            args.fusion = f_spec
        

    splits, valid_inds = get_inds(args)
    for i, (train_i, test_i) in enumerate(splits):
        train_loader, valid_loader, test_loader = get_loaders(args, train_i, test_i)

        run_setup(args, model_constructor, train_loader, valid_loader, test_loader,
                  run_name = " - ".join([f"Split {i}", f"smaller, 64e", f"{args.label_col}", f"{args.model}"]))
                
    
if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True

    main(args)