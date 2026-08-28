import argparse
from collections import defaultdict

import pandas as pd

from new_kf import *

def all_args():
    arg = defaultdict(lambda: {})
    arg["mix"]["tri"] = '--disable_wandb --mix_data --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang'.split()

    #bimodal
    arg["mix"]["pr"] = '--disable_wandb --mix_data --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --path_lang'.split()
    arg["mix"]["pc"] = '--disable_wandb --mix_data --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --path_lang --clinical'.split()
    arg["mix"]["rc"] = '--disable_wandb --mix_data --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --clinical'.split()

    #unimodal
    arg["mix"]["p"] = '--disable_wandb --mix_data --modality uni_path --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768'.split()
    arg["mix"]["r"] = '--disable_wandb --mix_data --modality uni_rad --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768'.split()
    arg["mix"]["c"] = '--disable_wandb --mix_data --modality uni_clin --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768'.split()

    # trimodal
    arg["mix3"]["tri"] = '--disable_wandb --mix_data --survival_years 3 --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang'.split()

    # bimodal
    arg["mix3"]["pr"] = '--disable_wandb --mix_data --survival_years 3 --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --path_lang'.split()
    arg["mix3"]["pc"] = '--disable_wandb --mix_data --survival_years 3 --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --path_lang --clinical'.split()
    arg["mix3"]["rc"] = '--disable_wandb --mix_data --survival_years 3 --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --clinical'.split()

    # unimodal
    arg["mix3"]["p"] = '--disable_wandb --mix_data --survival_years 3 --modality uni_path --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768'.split()
    arg["mix3"]["r"] = '--disable_wandb --mix_data --survival_years 3 --modality uni_rad --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768'.split()
    arg["mix3"]["c"] = '--disable_wandb --mix_data --survival_years 3 --modality uni_clin --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768'.split()

    #trimodal
    arg["iso"]["tri"] = '--disable_wandb --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang'.split()

    #bimodal
    arg["iso"]["pr"] = '--disable_wandb --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --path_lang'.split()
    arg["iso"]["pc"] = '--disable_wandb --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --path_lang --clinical'.split()
    arg["iso"]["rc"] = '--disable_wandb --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --clinical'.split()

    #unimodal
    arg["iso"]["p"] = '--disable_wandb --modality uni_path --loss_fn "cox_nll" --batch_size 128 --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768'.split()
    arg["iso"]["r"] = '--disable_wandb --modality uni_rad --loss_fn "cox_nll" --batch_size 128 --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768'.split()
    arg["iso"]["c"] = '--disable_wandb --modality uni_clin --loss_fn "cox_nll" --batch_size 128 --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768'.split()

    return arg

def run_all():
    parser = get_args_parser()

    arg_set = all_args()
    tables = {}

    for setup, runs in arg_set.items():
        summ5 = pd.DataFrame(index=["AUC", "C-Index"])
        summB = pd.DataFrame(index=["AUC", "C-Index"])

        for _, args in runs.items():
            args = parser.parse_args(args)
            mod, res5, resB = main(args)
            summ5[mod] = res5
            summB[mod] = resB

        tables[setup] = [summ5, summB]

    with open('./out.md', 'a', encoding="utf-8") as f:
        for setup, t in tables.items():
            print(setup)
            print()
            print("5Fold CV")
            print(t[0].to_markdown())
            print("Bootstrap")
            print(t[1].to_markdown())

            f.write("\n")
            f.write(f"# {setup}")
            f.write("\n")
            f.write(f"## 5Fold CV")
            f.write(t[0].to_markdown())
            f.write("## Bootstrap")
            f.write(t[1].to_markdown())



if __name__ == '__main__':
    run_all()