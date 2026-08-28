import argparse
from collections import defaultdict

import pandas as pd

from new_kf import *

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

    parser.add_argument('--mix_data',           action="store_true", 
                        help="Combine and then shuffle the train and test data")
    parser.add_argument('--modality',           type=str, default="multi", choices=["multi", "uni_path", "uni_rad", "uni_clin"])

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
    parser.add_argument('--wb_proj',            type=str,   default="New KFold Validation")
    parser.add_argument('--run_name')
    parser.add_argument('--debug',              action="store_true")
    return parser

def all_args():
    arg = defaultdict(lambda: {})
    arg["mix"]["tri"] = '--disable_wandb --mix_data --survival_years 2 --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang --label_col $target'.split(" ")

    #bimodal
    arg["mix"]["pr"] = '--disable_wandb --mix_data --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --path_lang --label_col $target'.split(" ")
    arg["mix"]["pc"] = '--disable_wandb --mix_data --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --path_lang --clinical --label_col $target'.split(" ")
    arg["mix"]["rc"] = '--disable_wandb --mix_data --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --clinical --label_col $target'.split(" ")

    #unimodal
    arg["mix"]["p"] = '--disable_wandb --mix_data --modality uni_path --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target'.split(" ")
    arg["mix"]["r"] = '--disable_wandb --mix_data --modality uni_rad --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target'.split(" ")
    arg["mix"]["c"] = '--disable_wandb --mix_data --modality uni_clin --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target'.split(" ")


    # trimodal
    arg["mix3"]["tri"] = '--disable_wandb --mix_data --survival_years 3 --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang --label_col $target'.split(" ")

    # bimodal
    arg["mix3"]["pr"] = '--disable_wandb --mix_data --survival_years 3 --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --path_lang --label_col $target'.split(" ")
    arg["mix3"]["pc"] = '--disable_wandb --mix_data --survival_years 3 --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --path_lang --clinical --label_col $target'.split(" ")
    arg["mix3"]["rc"] = '--disable_wandb --mix_data --survival_years 3 --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --clinical --label_col $target'.split(" ")

    # unimodal
    arg["mix3"]["p"] = '--disable_wandb --mix_data --survival_years 3 --modality uni_path --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target'.split(" ")
    arg["mix3"]["r"] = '--disable_wandb --mix_data --survival_years 3 --modality uni_rad --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target'.split(" ")
    arg["mix3"]["c"] = '--disable_wandb --mix_data --survival_years 3 --modality uni_clin --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target'.split(" ")

    #trimodal
    arg["iso"]["tri"] = '--disable_wandb --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang --label_col $target'.split(" ")

    #bimodal
    arg["iso"]["pr"] = '--disable_wandb --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --path_lang --label_col $target'.split(" ")
    arg["iso"]["pc"] = '--disable_wandb --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --path_lang --clinical --label_col $target'.split(" ")
    arg["iso"]["rc"] = '--disable_wandb --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --clinical --label_col $target'.split(" ")

    #unimodal
    arg["iso"]["p"] = '--disable_wandb --modality uni_path --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target'.split(" ")
    arg["iso"]["r"] = '--disable_wandb --modality uni_rad --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target'.split(" ")
    arg["iso"]["c"] = '--disable_wandb --modality uni_clin --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target'.split(" ")

    return arg

def run_all():
    parser = get_args_parser()

    arg_set = all_args()
    tables = {}

    for setup, runs in arg_set.items():
        summ5 = pd.DataFrame(index=["AUC", "C-Index"])
        summB = pd.DataFrame(index=["AUC", "C-Index"])

        for args in runs:
            args = parser.parse_args(args)
            mod, res5, resB = main(args)
            summ5[mod] = res5
            summB[mod] = resB

        tables[setup] = [summ5, summB]

    for setup, t in tables.items():
        print(setup)
        print()
        print("5Fold CV")
        print(t[0])
        print("Bootstrap")
        print(t[1])


if __name__ == '__main__':
    run_all()