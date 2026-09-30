import pandas as pd
import matplotlib.pyplot as plt

import new_kf
import km_stage

def get2():
    sub = {}
    sub[f"mix2"] = {
        "tri": f'--disable_wandb --mix_data --clinical --path_lang --rad_lang --stratify --emb_dim 8 --fusion_dim 64 16'.split() + \
                '--layer_norm --const_lr --modality hat --early -142 --plot'.split(),
        "pr": f'--disable_wandb --mix_data --rad_lang --path_lang --stratify --emb_dim 8 --fusion_dim 64 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -105'.split(),
        "pc": f'--disable_wandb --mix_data --path_lang --clinical --stratify --emb_dim 8 --fusion_dim 32 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -49'.split(),
        "rc": f'--disable_wandb --mix_data --rad_lang --clinical --stratify --emb_dim 16 --fusion_dim 64 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -31'.split(),
        "p": f'--disable_wandb --mix_data --modality uni_path --stratify --batch_size 256 --fusion_dim 128 32 --early -95'.split(),
        "r": f'--disable_wandb --mix_data --modality uni_rad --stratify --fusion_dim 128 32 16 --early -59'.split(),
        "c": f'--disable_wandb --mix_data --modality uni_clin --stratify --early -167'.split(),
        "km": '--mix_data'.split(),
    }

    sub[f"iso2"] = {
        "tri": f'--disable_wandb --clinical --path_lang --rad_lang --stratify --emb_dim 16 --fusion_dim 128 16'.split() + \
                '--layer_norm --const_lr --modality hat --early -93 --plot'.split(),
        "pr": f'--disable_wandb --rad_lang --path_lang --stratify --emb_dim 8 --fusion_dim 64 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -20'.split(),
        "pc": f'--disable_wandb --path_lang --clinical --stratify --emb_dim 16 --fusion_dim 128 32 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -112'.split(),
        "rc": f'--disable_wandb --rad_lang --clinical --stratify --emb_dim 16 --fusion_dim 128 16'.split() + \
                '--layer_norm --const_lr --modality hat --early -7 --plot'.split(),
        "p": f'--disable_wandb --modality uni_path --stratify --batch_size 256 --fusion_dim 8 --early -20'.split(),
        "r": f'--disable_wandb --modality uni_rad --stratify --fusion_dim 8 --early -86'.split(),
        "c": f'--disable_wandb --modality uni_clin --stratify --fusion_dim 32 --early -167'.split(),
        "km": [],
    }
    return sub

def get3():
    sub = {}
    sub[f"mix3"] = {
        "tri": f'--disable_wandb --survival_years 3 --mix_data --clinical --path_lang --rad_lang --stratify --emb_dim 8 --fusion_dim 128 16 8'.split() + \
                '--layer_norm --const_lr --modality hat --early -39 --plot'.split(),
        "pr": f'--disable_wandb --survival_years 3 --mix_data --rad_lang --path_lang --stratify --emb_dim 32 --fusion_dim 128 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -137'.split(),
        "pc": f'--disable_wandb --survival_years 3 --mix_data --path_lang --clinical --stratify --emb_dim 16 --fusion_dim 128 16 8'.split() + \
                '--layer_norm --const_lr --modality attn --early -44'.split(), 
        "rc": f'--disable_wandb --survival_years 3 --mix_data --rad_lang --clinical --stratify --emb_dim 16 --fusion_dim 128 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -78'.split(),
        "p": f'--disable_wandb --survival_years 3 --mix_data --modality uni_path --stratify --layer_norm --const_lr --batch_size 256 --fusion_dim 512 --early -93'.split(),
        "r": f'--disable_wandb --survival_years 3 --mix_data --modality uni_rad --stratify --layer_norm --const_lr --fusion_dim 128 --early -84'.split(),
        "c": f'--disable_wandb --survival_years 3 --mix_data --modality uni_clin --stratify --layer_norm --const_lr --early -32'.split(),
        "km": '--survival_years 3 --mix_data'.split()
    }

    sub[f"iso3"] = {
        "tri": f'--disable_wandb --survival_years 3 --clinical --path_lang --rad_lang --stratify --emb_dim 16 --fusion_dim 128 16'.split() + \
                '--layer_norm --const_lr --modality hat --early -72 --plot'.split(), #-> double check
        "pr": f'--disable_wandb --survival_years 3 --rad_lang --path_lang --stratify --emb_dim 8 --fusion_dim 64 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -20'.split(),
        "pc": f'--disable_wandb --survival_years 3 --path_lang --clinical --stratify --emb_dim 16 --fusion_dim 64 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -166'.split(),
        "rc": f'--disable_wandb --survival_years 3 --rad_lang --clinical --stratify --emb_dim 8 --fusion_dim 128 8'.split() + \
                '--layer_norm --const_lr --modality attn --early -131'.split(), 
        "p": f'--disable_wandb --survival_years 3 --modality uni_path --stratify --const_lr --fusion_dim 64 8 --early -165'.split(),
        "r": f'--disable_wandb --survival_years 3 --modality uni_rad --stratify --const_lr --fusion_dim 32 8 --early -161'.split(),
        "c": f'--disable_wandb --survival_years 3 --modality uni_clin --stratify --const_lr --fusion_dim 32 4 --early -159'.split(),
        "km": '--survival_years 3'.split(),
    }
    return sub

def get5():
    sub = {}
    sub[f"mix5"] = {
        "tri": f'--disable_wandb --survival_years 5 --mix_data --clinical --path_lang --rad_lang --stratify --emb_dim 8 --fusion_dim 64 16'.split() + \
                '--layer_norm --const_lr --modality hat --early -86 --plot'.split(),
        "pr": f'--disable_wandb --survival_years 5 --mix_data --rad_lang --path_lang --stratify --emb_dim 16 --fusion_dim 64 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -32'.split(),
        "pc": f'--disable_wandb --survival_years 5 --mix_data --path_lang --clinical --stratify --emb_dim 16 --fusion_dim 64 16 8'.split() + \
                '--layer_norm --const_lr --modality attn --early -100'.split(),
        "rc": f'--disable_wandb --survival_years 5 --mix_data --rad_lang --clinical --stratify --emb_dim 16 --fusion_dim 64 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -157'.split(),
        "p": f'--disable_wandb --survival_years 5 --mix_data --modality uni_path --stratify --const_lr --batch_size 256 --fusion_dim 512 64 16 --early -142'.split(),
        "r": f'--disable_wandb --survival_years 5 --mix_data --modality uni_rad --stratify --const_lr --fusion_dim 64 16 --early -55'.split(),
        "c": f'--disable_wandb --survival_years 5 --mix_data --modality uni_clin --stratify --const_lr --fusion_dim 32 4 --early -82'.split(),
        "km": '--survival_years 5 --mix_data'.split()
    }

    sub[f"iso5"] = {
        "tri": f'--disable_wandb --survival_years 5 --clinical --path_lang --rad_lang --stratify --emb_dim 16 --fusion_dim 128 16'.split() + \
                '--layer_norm --const_lr --modality hat --early -93 --plot'.split(),
        "pr": f'--disable_wandb --survival_years 5 --rad_lang --path_lang --stratify --emb_dim 8 --fusion_dim 64 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -20'.split(),
        "pc": f'--disable_wandb --survival_years 5 --path_lang --clinical --stratify --emb_dim 16 --fusion_dim 128 32 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -4'.split(),
        "rc": f'--disable_wandb --survival_years 5 --rad_lang --clinical --stratify --emb_dim 16 --fusion_dim 128 16'.split() + \
                '--layer_norm --const_lr --modality attn --early -7'.split(),
        "p": f'--disable_wandb --survival_years 5 --modality uni_path --stratify --const_lr --fusion_dim 64 8 --early -166'.split(),
        "r": f'--disable_wandb --survival_years 5 --modality uni_rad --stratify --const_lr --fusion_dim 64 8 --early -124'.split(),
        "c": f'--disable_wandb --survival_years 5 --modality uni_clin --stratify --const_lr --fusion_dim 32 4 --early -166'.split(),
        "km": '--survival_years 5'.split(),
    }
    return sub

def all_args():
    arg = {
        **get2(),
        **get3(),
        **get5(),
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
    plt.tight_layout() 
    plt.savefig(f"vises/intervals/{setup}_{met}_intervals.jpeg")
    plt.close()

def plot_plotly(setup, df, met):
    idx = pd.IndexSlice
    means = df.loc[met, idx[:, "Mean"]].to_numpy()
    lows = means - df.loc[met, idx[:, "CI Low"]].to_numpy()
    highs = df.loc[met, idx[:, "CI High"]].to_numpy() - means
    
    import plotly.express as px

    fig = px.scatter(
        x=means,
        y=df.columns.get_level_values(0).unique(),
        error_x=highs, 
        error_x_minus=lows, 
        title=f"Summary of {met} Performance on {setup}"
    )

    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text="Model")
    fig.write_image(f"vises/intervals/{setup}_{met}_intervals.pdf")


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

    with open('vises/out.md', 'w', encoding="utf-8") as f:
        for setup, t in tables.items():
            plot_plotly(setup, t[0], "AUC")
            plot_plotly(setup, t[0], "C-Index")

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

            with pd.ExcelWriter('vises/res.xlsx', engine='openpyxl') as writer:
                t[0].to_excel(writer, sheet_name=f'{setup}')


if __name__ == '__main__':
    run_all()