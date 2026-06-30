import os
import argparse

import numpy as np
from torch import nn
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split, KFold

from data import *
from model import create_mlp, EmbMIL, DenseFusion
from util import *
from run import *

def get_args_parser():
    parser = argparse.ArgumentParser('Supervised Panc Training', add_help=False)

    parser.add_argument('--seed',               type=int,   default=0)
      
    parser.add_argument('--batch_size',         type=int,   default=16)
    parser.add_argument('--loss_fn',            type=str,   default="bce", choices=list(LOSS_FNS.keys()))
    parser.add_argument('--model',              type=str,   default="conch", choices=['conch', 'biomedclip', 'gemma', 'qwen'])
    # parser.add_argument('--data_path',          type=str,   default="../{model}_path_rad_text_embs")
    parser.add_argument('--data_path',          type=str,   default="../updated_multimodal_bins")
    parser.add_argument('--test_path',          type=str,   default="../multimodal_bins_rw")
    parser.add_argument('--epochs',             type=int,   default=200)
    parser.add_argument('--device',                         default='cuda')
    parser.add_argument('--float16',            type=bool,  default=True)
    parser.add_argument('--early_stop',         type=bool,  default=True)
    parser.add_argument('--patience',           type=int,   default=5)
    parser.add_argument('--folds',              type=int,   default=5)

    parser.add_argument('--mix_data',           action="store_true")
    parser.add_argument('--sparse',             action="store_true")
    parser.add_argument('--clinical',           action="store_true")
    parser.add_argument('--clinical_imputed',   action="store_true")
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
    parser.add_argument('--wb_proj',            type=str,   default="KFold Validation")
    parser.add_argument('--run_name')
    parser.add_argument('--debug',              action="store_true")
    return parser

# --------------------------------------------------------

def get_clinical_encoder(args):
    # clin_enc = create_mlp(24, [128], args.emb_dim, act = nn.GELU(), dropout = 0.3, layer_norm = True)
    clin_enc = create_mlp(24, [], args.emb_dim, act = nn.GELU(), dropout = 0.3, end_with_fc=False, end_with_dropout=True, rms_norm = True, end_with_norm=True)
    return clin_enc, False

def get_path_lang_encoder(args):
    # path_lang_enc = create_mlp(args.enc_dim, [128], args.emb_dim, act = nn.GELU(), dropout = 0.4, layer_norm = True)
    path_lang_enc = create_mlp(args.enc_dim, [], args.emb_dim, act = nn.GELU(), dropout = 0.3, end_with_fc=False, end_with_dropout=True, rms_norm = True, end_with_norm=True)
    return path_lang_enc, True

def get_rad_lang_encoder(args):
    # rad_lang_enc = create_mlp(args.enc_dim, [128], args.emb_dim, act = nn.GELU(), dropout = 0.4, layer_norm = True)
    rad_lang_enc = create_mlp(args.enc_dim, [], args.emb_dim, act = nn.GELU(), dropout = 0.3, end_with_fc=False, end_with_dropout=True, rms_norm = True, end_with_norm=True)
    return rad_lang_enc, True

def get_path_img_encoder(args):
    mil = EmbMIL(embed_dim=384, dropout=0.3, attn_dim = 256, proj_dim=args.emb_dim, loss_fn=None)
    return mil, True

def get_dense_fusion_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = DenseFusion(encs, emb_dim=args.emb_dim, hidden_dims=[32], autocast=casts, loss_fn=LOSS_FNS[args.loss_fn], device=device)
    return model, device

# --------------------------------------------------------

def main(args):
    args.wb_proj = f"{args.model} {args.wb_proj} {args.folds}"
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    datasets = get_datasets(args)
    dataset = ConcatDataset(datasets) #this is prob a little mem inefficient

    tv, test_inds = train_test_split(
        np.arange(len(dataset)),
        test_size=0.08,
        random_state=args.seed,
        #consider stratifying
    )


    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = kf.split(X=tv)

    for split, (train_inds, val_inds) in enumerate(splits):
        train_loader, valid_loader, test_loader = get_combined_loaders(args, dataset, train_inds, val_inds, test_inds)

        run_name = f"Split {split} "

        #NOTE: THE MODEL IS TRAINED AS LOG HAZARD PREDICTING
        run_setup(args, get_dense_fusion_model, train_loader, valid_loader, test_loader, run_name = run_name)
       

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True

    main(args)