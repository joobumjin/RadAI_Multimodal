import pandas as pd
import matplotlib.pyplot as plt

import new_kf
import km_stage

def get_sub(yrs: int):
    sub = {}
    sub[f"mix{yrs}"] = {
        #trimodal
        "tri": f'--disable_wandb --survival_years {yrs} --mix_data --clinical --path_lang --rad_lang --stratify --emb_dim 8 --fusion_dim 64 16 --epochs 500 --layer_norm --const_lr --lr 1e-3 --disable_wandb --batch_size 128 --modality hat --early -142'.split(),
        #bimodal
        "pr": f'--disable_wandb --survival_years {yrs} --mix_data --rad_lang --path_lang --stratify --emb_dim 128 --modality attn'.split(),
        "pc": f'--disable_wandb --survival_years {yrs} --mix_data --path_lang --clinical --stratify --emb_dim 128 --modality attn'.split(),
        "rc": f'--disable_wandb --survival_years {yrs} --mix_data --rad_lang --clinical --stratify --emb_dim 128 --modality attn'.split(),
        #unimodal
        "p": f'--disable_wandb --survival_years {yrs} --mix_data --modality uni_path --stratify'.split(),
        "r": f'--disable_wandb --survival_years {yrs} --mix_data --modality uni_rad --stratify'.split(),
        "c": f'--disable_wandb --survival_years {yrs} --mix_data --modality uni_clin --stratify'.split(),
        #km stage
        "km": f'--survival_years {yrs} --mix_data'.split(),
    }

    sub[f"iso{yrs}"] = {
        #trimodal
        "tri": f'--disable_wandb --survival_years {yrs} --clinical --path_lang --rad_lang'.split(),
        #bimodal
        "pr": f'--disable_wandb --survival_years {yrs} --rad_lang --path_lang'.split(),
        "pc": f'--disable_wandb --survival_years {yrs} --path_lang --clinical'.split(),
        "rc": f'--disable_wandb --survival_years {yrs} --rad_lang --clinical'.split(),
        #unimodal
        "p": f'--disable_wandb --survival_years {yrs} --modality uni_path '.split(),
        "r": f'--disable_wandb --survival_years {yrs} --modality uni_rad'.split(),
        "c": f'--disable_wandb --survival_years {yrs} --modality uni_clin '.split(),
        #km stage
        "km": f'--survival_years {yrs}'.split(),
    }
    return sub

def all_args():
    arg = {
        **get_sub(2),
        **get_sub(3),
        **get_sub(5),
    }

    return arg

def plot(setup, df, met):
    idx = pd.IndexSlice
    means = df.loc[met, idx[:, "Mean"]].to_numpy()
    lows = means - df.loc[met, idx[:, "CI Low"]].to_numpy()
    highs = df.loc[met, idx[:, "CI High"]].to_numpy() - means
    
    plt.errorbar(means, df.columns.get_level_values(0).unique(), xerr=[lows, highs], fmt='o', elinewidth=2, capsize=6, 
                 label='95% CI')
    plt.title(f"Summary of {met} Performance on {setup}")
    plt.savefig(f"./{setup}_{met}_intervals.jpeg")
    plt.close()

def run_all():
    arg_set = all_args()
    tables, text = {}, {} #{setup: [summary_tables]}

    for setup, runs in arg_set.items(): #{setup: runs ({model_type: args})}
        models = ["Trimodal", "Bimodal Path+Rad", "Bimodal Path+Clin", "Bimodal Rad+Clin", "Unimodal Path", "Unimodal Rad", "Unimodal Clin", "KM Stage"]
        cols = []
        for m in models:
            cols += [(m, "Mean"), (m, "CI Low"), (m, "CI High")]

        summ5 = pd.DataFrame(index=["AUC", "C-Index"], columns=cols)
        summ5.index.name = "Metric (95% CI)"
        summ5.columns = pd.MultiIndex.from_tuples(summ5.columns)

        text5 = pd.DataFrame(index=["AUC", "C-Index"], columns=models)

        for n, args in runs.items():
            if n != "km": 
                parser = new_kf.get_args_parser()
                main_fn = new_kf.main

            else:
                parser = km_stage.get_args_parser()
                main_fn = km_stage.main

            args = parser.parse_args(args)
            mod, res5, _ = main_fn(args)
            summ5[(mod, "Mean")] = [r.avg for r in res5]
            summ5[(mod, "CI Low")] = [r.ci[0] for r in res5]
            summ5[(mod, "CI High")] = [r.ci[1] for r in res5]

            text5[mod] = [f"{r.avg:.3f} ({r.ci[0]:.3f}, {r.ci[1]:.3f})" for r in res5]

        tables[setup] = [summ5]
        text[setup] = text5

    with open('./out.md', 'w', encoding="utf-8") as f:
        for setup, t in tables.items():
            plot(setup, t[0], "AUC")
            plot(setup, t[0], "C-Index")

            print("\n--------")
            print(setup)
            print()
            print("5 Fold CV")
            print(text[setup])

            f.write("\n")
            f.write(f"# {setup}")
            f.write("\n")
            f.write("## 5 Fold CV")
            f.write("\n")
            f.write(t[0].to_markdown())

if __name__ == '__main__':
    run_all()