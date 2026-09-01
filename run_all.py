from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import new_kf
import km_stage

def get_sub(yrs: int):
    sub = {}
    sub[f"mix{yrs}"] = {
        #trimodal
        "tri": f'--disable_wandb--survival_years {yrs} --mix_data --clinical --path_lang --rad_lang --plot'.split(),
        #bimodal
        "pr": f'--disable_wandb--survival_years {yrs} --mix_data --rad_lang --path_lang'.split(),
        "pc": f'--disable_wandb--survival_years {yrs} --mix_data --path_lang --clinical'.split(),
        "rc": f'--disable_wandb--survival_years {yrs} --mix_data --rad_lang --clinical'.split(),
        #unimodal
        "p": f'--disable_wandb--survival_years {yrs} --mix_data --modality uni_path'.split(),
        "r": f'--disable_wandb--survival_years {yrs} --mix_data --modality uni_rad'.split(),
        "c": f'--disable_wandb--survival_years {yrs} --mix_data --modality uni_clin'.split(),
        #km stage
        "km": f'--survival_years {yrs} --mix_data'.split(),
    }

    sub[f"iso{yrs}"] = {
        #trimodal
        "tri": f'--disable_wandb--survival_years {yrs} --clinical --path_lang --rad_lang'.split(),
        #bimodal
        "pr": f'--disable_wandb--survival_years {yrs} --rad_lang --path_lang'.split(),
        "pc": f'--disable_wandb--survival_years {yrs} --path_lang --clinical'.split(),
        "rc": f'--disable_wandb--survival_years {yrs} --rad_lang --clinical'.split(),
        #unimodal
        "p": f'--disable_wandb --survival_years {yrs} --modality uni_path '.split(),
        "r": f'--disable_wandb --survival_years {yrs} --modality uni_rad'.split(),
        "c": f'--disable_wandb --survival_years {yrs} --modality uni_clin '.split(),
        #km stage
        "km": f'--survival_years {yrs}'.split(),
    }
    return sub

def get_strat(yrs: int):
    sub = {}

    sub[f"strat{yrs}"] = {
        #trimodal
        "tri": f'--disable_wandb --survival_years {yrs} --mix_data --clinical --path_lang --rad_lang --stratify'.split(),
        #bimodal
        "pr": f'--disable_wandb --survival_years {yrs} --mix_data --rad_lang --path_lang --stratify'.split(),
        "pc": f'--disable_wandb --survival_years {yrs} --mix_data --path_lang --clinical --stratify'.split(),
        "rc": f'--disable_wandb --survival_years {yrs} --mix_data --rad_lang --clinical --stratify'.split(),
        #unimodal
        "p": f'--disable_wandb --survival_years {yrs} --mix_data --modality uni_path --stratify'.split(),
        "r": f'--disable_wandb --survival_years {yrs} --mix_data --modality uni_rad --stratify'.split(),
        "c": f'--disable_wandb --survival_years {yrs} --mix_data --modality uni_clin --stratify'.split(),
        #km stage
        "km": f'--stratify --survival_years {yrs}'.split(),
    }

    return sub

def all_args():
    arg = {
        **get_sub(2),
        **get_strat(2),
        **get_sub(3),
        **get_sub(5),
    }

    return arg

def plot(setup, df, met):
    idx = pd.IndexSlice
    means = df[met, idx[:, "Mean"]].to_numpy()
    lows = means - df[met, idx[:, "CI Low"]].to_numpy()
    highs = df[met, idx[:, "CI High"]].to_numpy() - means
    
    plt.errorbar(means, df.columns.get_level_values(0).unique(), xerr=[lows, highs], fmt='o', elinewidth=2, capsize=6, 
                 label='95% CI')
    plt.savefig(f"./{setup}_intervals.jpeg")

def run_all():
    arg_set = all_args()
    tables = {} #{setup: [summary_tables]}

    for setup, runs in arg_set.items(): #{setup: runs ({model_type: args})}
        models = ["Trimodal", "Bimodal Path+Rad", "Bimodal Path+Clin", "Bimodal Rad+Clin", "Unimodal Path", "Unimodal Rad", "Unimodal Clin", "KM Stage"]
        cols = []
        for m in models:
            cols += [(m, "Mean"), (m, "CI Low"), (m, "CI High")]

        summB = pd.DataFrame(index=["AUC", "C-Index"], columns=cols)
        summB.index.name = "Metric (95% CI)"

        for n, args in runs.items():
            if n != "km": 
                parser = new_kf.get_args_parser()
                main_fn = new_kf.main

            else:
                parser = km_stage.get_args_parser()
                main_fn = km_stage.main

            args = parser.parse_args(args)
            mod, _, resB = main_fn(args)
            summB[(mod, "Mean")] = [r.avg for r in resB]
            summB[(mod, "CI Low")] = [r.ci[0] for r in resB]
            summB[(mod, "CI High")] = [r.ci[1] for r in resB]

        tables[setup] = [summB]

    with open('./out.md', 'w', encoding="utf-8") as f:
        for setup, t in tables.items():
            print("\n--------")
            print(setup)
            print()
            print("Bootstrap")
            print(t[0])

            f.write("\n")
            f.write(f"# {setup}")
            f.write("\n")
            f.write("## Bootstrap")
            f.write(t[0].to_markdown())

if __name__ == '__main__':
    run_all()