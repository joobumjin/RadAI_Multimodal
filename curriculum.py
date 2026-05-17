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
    parser.add_argument('--model',              type=str,   default="conch", choices=['conch', 'biomedclip'])
    # parser.add_argument('--data_path',          type=str,   default="../{model}_path_rad_text_embs")
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
    parser.add_argument('--clinical_imputed',   action="store_true")
    parser.add_argument('--path_lang',          action="store_true")
    parser.add_argument('--rad_lang',           action="store_true")
    parser.add_argument('--path_img',           action="store_true")
    parser.add_argument('--emb_dim',            type=int,   default=64)
    
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
    parser.add_argument('--wb_proj',            type=str,   default="Panc MM External Test")
    parser.add_argument('--debug',              action="store_true")
    return parser

# --------------------------------------------------------

def get_inds(args):
    keys = ["slide_ids", "survival_days", "survival_right_censor"]

    index   = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)
    inds    = np.arange(index['survival_days'])

    label_mask = ~np.isnan(index[args.label_col])
    exclusion_mask = ~index["excluded"]
    mask = label_mask & exclusion_mask
    if "indicator" in args.label_col: 
        for key in keys:
            mask = mask & (~np.isnan(index[key]))

    modality_mask = np.zeros_like(mask).astype(bool) if args.sparse else np.ones_like(mask).astype(bool)
    combine_op = lambda x, y: x | y if args.sparse else x & y

    arg_dict = vars(args)
    for mod in ["clinical", "clinical_imputed"]:
        if arg_dict.get(mod, False): 
            modality_mask = combine_op(modality_mask, ~index[f'{mod}_mask'])
    for mod in ["path_lang", "rad_lang", "path_img"]:
        if arg_dict.get(mod, False): 
            modality_mask = combine_op(modality_mask, ~(index[f'{mod}_mask']))

    mask = mask & modality_mask

    valid_inds = inds[mask]
    num_train = int(len(valid_inds) * args.train_split)
    if not args.clinical_imputed: #standard random shuffle and select
        np.random.shuffle(valid_inds)
        
        train_inds, validation_inds = valid_inds[:num_train], valid_inds[num_train:]
    else: #only test on real samples, but can train on imputed data
        orig = inds[~np.isnan(index['clinical']).any(axis=1) & mask]
        np.random.shuffle(orig)

        imputed = inds[np.isnan(index['clinical']).any(axis=1) & mask]

        num_test = len(inds) - num_train
        train_inds, validation_inds = np.hstack((orig[num_test:], imputed)), orig[:num_test]

    return train_inds, validation_inds

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

def get_dense_fusion_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    losses = {
        "l1": F.l1_loss,
        "smooth_l1": F.smooth_l1_loss,
        "mse": F.mse_loss,
        "bce": F.binary_cross_entropy_with_logits,
    }

    fusers = {
        "naive_sum": NaiveSum,
        "naive_avg": NaiveAvg,
        "weighted_sum": LearnedWeightSum
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

    model = EmbFusion(encs, emb_dim=args.emb_dim, hidden_dims=[32], autocast=casts, fusion_fn=fusers[args.fusion], loss_fn=losses[args.loss_fn], device=device)
    return model, device

# --------------------------------------------------------

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, valid_loader, test_loader = get_loaders(args, *get_inds(args))

    ## reconstruction loss, recurrence regression loss, survival regression loss?, survival binary loss

    run_setup(args, get_dense_fusion_model, train_loader, valid_loader, test_loader, 
              run_name = " - ".join([f"smaller, 64e", f"{args.label_col}", f"{args.model}"]))
       

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True

    main(args)