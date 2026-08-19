import os
import argparse

import numpy as np
from torch import nn
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split

from data import *
from util import *
from run import *
from final_models import *

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

    parser.add_argument('--mix_data',           action="store_true", 
                        help="Combine and then shuffle the train and test data")
    parser.add_argument('--modality',           type=str, default="multi", choices=["multi", "uni_path", "uni_rad", "uni_clin"])
    
    parser.add_argument('--sparse',             action="store_true")
    parser.add_argument('--multi',              action="store_true")
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
    parser.add_argument('--wb_proj',            type=str,   default="Panc MM Examine")
    parser.add_argument('--run_name')
    parser.add_argument('--debug',              action="store_true")
    return parser

# --------------------------------------------------------

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.mix_data:
        datasets = get_datasets(args)
        dataset = ConcatDataset(datasets) #this is prob a little mem inefficient

        tv, test_inds = train_test_split(
            np.arange(len(dataset)),
            test_size=0.08,
            random_state=args.seed,
            #consider stratifying
        )

        train_inds, val_inds = train_test_split(
            tv,
            test_size=0.08,
            random_state=args.seed
        )
        # train_inds, val_inds = tv, []

        train_loader, valid_loader, test_loader = get_combined_loaders(args, dataset, train_inds, val_inds, test_inds)

    else:
        tv_set, test_set = get_datasets(args)
        tv = np.arange(len(tv_set))

        train_inds, val_inds = train_test_split(
            tv,
            test_size=1-args.train_split,
            random_state=args.seed,
        )

        train_loader, valid_loader, test_loader = get_subset_loaders(args, tv_set, train_inds, val_inds, test_set)

    run_name = args.run_name if args.run_name is not None else " - ".join([f"{args.emb_dim}e{", sparse" if args.sparse else ""}", f"{args.label_col}", f"{args.model} - "])

    model_fns = {
        "multi": get_dense_fusion_model,
        "uni_path": get_unimodal_path,
        "uni_rad": get_unimodal_rad,
        "uni_clin": get_unimodal_clin
    }

    #NOTE: THE MODEL IS TRAINED AS LOG HAZARD PREDICTING
    run_setup(args, model_fns[args.modality], train_loader, valid_loader, test_loader, run_name = run_name)
       

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True

    if args.modality == "uni_path": args.path_lang = True
    elif args.modality == "uni_rad": args.rad_lang = True
    elif args.modality == "uni_clin": args.clinical = True

    main(args)