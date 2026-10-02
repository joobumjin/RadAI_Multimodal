from itertools import chain

import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

import new_kf
import km_stage

DEFAULTS = "--layer_norm --const_lr --disable_wandb --stratify".split(" ")

def get2():
    sub = {}

    sub[f"mix2"] = {
        "tri": f'--clinical --path_lang --rad_lang --emb_dim 8 --fusion_dim 64 16'.split() + \
            '--modality hat --early -142 --plot'.split(),
        "pr": f'--rad_lang --path_lang --emb_dim 8 --fusion_dim 64 16'.split() + \
            '--modality attn --early -105'.split(),
        "pc": f'--path_lang --clinical --emb_dim 16 --fusion_dim 64 16'.split() + \
            '--modality attn --early -54'.split(),
        "rc": f'--rad_lang --clinical --emb_dim 16 --fusion_dim 64 16'.split() + \
            '--modality attn --early -31'.split(),
        "p": f'--modality uni_path --batch_size 256 --fusion_dim 128 32 --early -95'.split(),
        "r": f'--modality uni_rad --fusion_dim 128 32 16 --early -59'.split(),
        "c": f'--modality uni_clin --early -167'.split(),
        "km": [],
    }

    sub["mix2"] = {setup: args + ['--mix_data'] for setup, args in sub["mix2"].items()}

    sub[f"iso2"] = {
        "tri": f'--clinical --path_lang --rad_lang --emb_dim 16 --fusion_dim 128 16'.split() + \
            '--modality hat --early -93 --plot'.split(),
        "pr": f'--rad_lang --path_lang --emb_dim 8 --fusion_dim 64 16'.split() + \
            '--modality attn --early -20'.split(),
        "pc": f'--path_lang --clinical --emb_dim 16 --fusion_dim 128 32 16'.split() + \
            '--modality attn --early -112'.split(),
        "rc": f'--rad_lang --clinical --emb_dim 16 --fusion_dim 128 16'.split() + \
            '--modality hat --early -44'.split(),
        "p": f'--modality uni_path --batch_size 256 --fusion_dim 8 --early -20'.split(),
        "r": f'--modality uni_rad --fusion_dim 8 --early -86'.split(),
        "c": f'--modality uni_clin --fusion_dim 32 --early -167'.split(),
        "km": [],
    }

    return sub

def get3():
    sub = {}

    sub[f"mix3"] = {
        "tri": f'--clinical --path_lang --rad_lang --emb_dim 8 --fusion_dim 128 16 8'.split() + \
            '--modality hat --early -39 --plot'.split(),
        "pr": f'--rad_lang --path_lang --emb_dim 32 --fusion_dim 128 16'.split() + \
            '--modality attn --early -137'.split(),
        "pc": f'--path_lang --clinical --emb_dim 16 --fusion_dim 128 16 8'.split() + \
            '--modality attn --early -44'.split(), 
        "rc": f'--rad_lang --clinical --emb_dim 16 --fusion_dim 128 16'.split() + \
            '--modality attn --early -78'.split(),
        "p": f'--modality uni_path --batch_size 256 --fusion_dim 512 --early -93'.split(),
        "r": f'--modality uni_rad --fusion_dim 128 --early -84'.split(),
        "c": f'--modality uni_clin --early -32'.split(),
        "km": []
    }

    sub["mix3"] = {setup: args + ['--survival_years', '3', '--mix_data'] for setup, args in sub["mix3"].items()}

    sub[f"iso3"] = {
        "tri": f'--clinical --path_lang --rad_lang --emb_dim 16 --fusion_dim 256 8'.split() + \
            '--modality hat --early -54 --plot'.split(), 
        "pr": f'--rad_lang --path_lang --emb_dim 8 --fusion_dim 64 16'.split() + \
            '--modality attn --early -20'.split(),
        "pc": f'--path_lang --clinical --emb_dim 16 --fusion_dim 64 16'.split() + \
            '--modality attn --early -166'.split(),
        "rc": f'--rad_lang --clinical --emb_dim 8 --fusion_dim 128 8'.split() + \
            '--modality attn --early -131'.split(), 
        "p": f'--modality uni_path --fusion_dim 64 8 --early -165'.split(),
        "r": f'--modality uni_rad --fusion_dim 32 8 --early -161'.split(),
        "c": f'--modality uni_clin --fusion_dim 32 4 --early -159'.split(),
        "km": [],
    }

    sub["iso3"] = {setup: args + ['--survival_years', '3'] for setup, args in sub["iso3"].items()}

    return sub

def get5():
    sub = {}
    
    sub[f"mix5"] = {
        "tri": f'--clinical --path_lang --rad_lang --emb_dim 8 --fusion_dim 64 16'.split() + \
            '--modality hat --early -86 --plot'.split(), #-> ?
        "pr": f'--rad_lang --path_lang --emb_dim 16 --fusion_dim 64 16'.split() + \
            '--modality attn --early -32'.split(),
        "pc": f'--path_lang --clinical --emb_dim 16 --fusion_dim 64 16 8'.split() + \
            '--modality attn --early -100'.split(),
        "rc": f'--rad_lang --clinical --emb_dim 16 --fusion_dim 64 16'.split() + \
            '--modality attn --early -157'.split(),
        "p": f'--modality uni_path --batch_size 256 --fusion_dim 512 64 16 --early -142'.split(),
        "r": f'--modality uni_rad --fusion_dim 64 16 --early -55'.split(),
        "c": f'--modality uni_clin --fusion_dim 32 4 --early -82'.split(),
        "km": []
    }

    sub["mix5"] = {setup: args + ['--survival_years', '5', '--mix_data'] for setup, args in sub["mix5"].items()}

    sub[f"iso5"] = {
        "tri": f'--clinical --path_lang --rad_lang --emb_dim 8 --fusion_dim 128 32 16'.split() + \
            '--modality hat --early -40 --plot'.split(),
        "pr": f'--rad_lang --path_lang --emb_dim 8 --fusion_dim 64 16'.split() + \
            '--modality attn --early -20'.split(),
        "pc": f'--path_lang --clinical --emb_dim 16 --fusion_dim 128 32 16'.split() + \
            '--modality attn --early -4'.split(),
        "rc": f'--rad_lang --clinical --emb_dim 16 --fusion_dim 128 16'.split() + \
            '--modality attn --early -7'.split(),
        "p": f'--modality uni_path --const_lr --fusion_dim 64 8 --early -166'.split(),
        "r": f'--modality uni_rad --const_lr --fusion_dim 64 8 --early -124'.split(),
        "c": f'--modality uni_clin --const_lr --fusion_dim 32 4 --early -166'.split(),
        "km": [],
    }

    sub["iso5"] = {setup: args + ['--survival_years', '5'] for setup, args in sub["iso5"].items()}
    
    return sub

def all_args():
    return {
        # "2yr": get2(),
        # "3yr": get3(),
        "5yr": get5(),
    }

def plot(setup, df, met):
    """
    Matplotlib interval plots on metric performance
    params:
        - setup: str - just a string describing experimental setup
        - df: pd.DataFrame - MultiIndex dataframe with metrics on row and [Model, ["Mean", "CI Low", "CI High"]] on multi-columns
        - met: str - which metric (row from df) to plot
    """
    idx = pd.IndexSlice
    means = df.loc[met, idx[:, "Mean"]].to_numpy()
    lows = means - df.loc[met, idx[:, "CI Low"]].to_numpy()
    highs = df.loc[met, idx[:, "CI High"]].to_numpy() - means
    
    plt.errorbar(means, df.columns.get_level_values(0).unique(), xerr=[lows, highs], fmt='o', elinewidth=2, capsize=6, label='95% CI')
    plt.title(f"Summary of {met} Performance on {setup}")
    plt.tight_layout() 
    plt.savefig(f"vises/intervals/{setup}_{met}_intervals.jpeg")
    plt.close()

def plot_plotly(setup, df, met):
    """
    Plotly interval plots on metric performance
    params:
        - setup: str - just a string describing experimental setup
        - df: pd.DataFrame - MultiIndex dataframe with metrics on row and [Model, ["Mean", "CI Low", "CI High"]] on multi-columns
        - met: str - which metric (row from df) to plot
    """
    idx = pd.IndexSlice
    means = df.loc[met, idx[:, "Mean"]].to_numpy()
    lows = means - df.loc[met, idx[:, "CI Low"]].to_numpy()
    highs = df.loc[met, idx[:, "CI High"]].to_numpy() - means
    
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

def combined_plotly(yr, tables):
    fig = make_subplots(rows=2, cols=len(tables), subplot_titles=("Mix AUC", "Mix C-Index", "Iso AUC", "Iso C-Index"))
    
    for setup_ind, (setup, table) in enumerate(tables.items()):
        for met_ind, met in zip([1, 2], ["AUC", "C-Index"]):
            idx = pd.IndexSlice
            means = table.loc[met, idx[:, "Mean"]].to_numpy()
            lows = means - table.loc[met, idx[:, "CI Low"]].to_numpy()
            highs = table.loc[met, idx[:, "CI High"]].to_numpy() - means

            sub = px.scatter(
                x=means,
                y=table.columns.get_level_values(0).unique(),
                error_x=highs, 
                error_x_minus=lows, 
                title=f"Summary of {met} Performance on {setup}"
            )
            for trace in sub.data:
                fig.add_trace(trace, row = met_ind, col=setup_ind+1)
            fig.update_xaxes(range=[0., 1.], row = met_ind, col=setup_ind+1)

    fig.update_layout(height=480, width=1400, title_text=f"{yr} Performance")
    fig.write_image(f"vises/intervals/{yr}_Summarized.pdf")


def run_all():
    """
    Runs all the experiments in all_args() and 
    plots model performance in each experimental setup
    """
    arg_set = all_args()
    models = ["Trimodal", "Bimodal Path+Rad", "Bimodal Path+Clin", "Bimodal Rad+Clin", 
              "Unimodal Path", "Unimodal Rad", "Unimodal Clin", "KM Stage"]
    
    cols = list(chain.from_iterable([
        [(m, "Mean"), (m, "CI Low"), (m, "CI High")] 
        for m in models
    ]))
    
    for yr, arg in arg_set.items():
        tables = {}

        for setup, runs in arg.items():
            summ5 = pd.DataFrame(index=["AUC", "C-Index"], columns=cols)
            summ5.index.name = "Metric (95% CI)"
            summ5.columns = pd.MultiIndex.from_tuples(summ5.columns)

            for model, args in runs.items():
                if model != "km": 
                    parser = new_kf.get_args_parser()
                    main_fn = new_kf.main
                    args += DEFAULTS

                else:
                    parser = km_stage.get_args_parser()
                    main_fn = km_stage.main

                args = parser.parse_args(args)
                mod, res5, _ = main_fn(args)
                summ5[(mod, "Mean")] = [r.avg for r in res5]
                summ5[(mod, "CI Low")] = [r.ci[0] for r in res5]
                summ5[(mod, "CI High")] = [r.ci[1] for r in res5]

            tables[setup] = summ5
            plot_plotly(setup, summ5, "AUC")
            plot_plotly(setup, summ5, "C-Index")

        combined_plotly(yr, tables)    
        with pd.ExcelWriter('vises/res.xlsx', engine='openpyxl', mode='a', if_sheet_exists="overlay") as writer:
            for i, (setup, table) in enumerate(tables.items()):
                table.to_excel(writer, sheet_name=yr, startrow=i*(len(table) + 3) + 2)

                worksheet = writer.sheets[yr]
                worksheet[f"A{i*(len(table) + 3) + 1}"] = setup

if __name__ == '__main__':
    run_all()