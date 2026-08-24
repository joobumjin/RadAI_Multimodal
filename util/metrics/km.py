import pandas as pd
import torch
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns

def km_preds(all_deaths, all_times, all_preds):
    sns.set_theme()

    # all_preds = torch.exp(all_preds)
    pred_bins = torch.linspace(torch.min(all_preds), torch.max(all_preds), 3)
    pred_binned = torch.bucketize(all_preds, pred_bins)

    df = pd.DataFrame({"Death": all_deaths, "Time": all_times, "Pred Bins": pred_binned})

    # Initialize the fitter
    kmf = KaplanMeierFitter()
    ax = plt.subplot(111)

    for name, grouped_df in df.groupby("Pred Bins"):
        kmf.fit(
            grouped_df["Time"],
            event_observed=grouped_df["Death"],
            label=name,
        )
        kmf.plot_survival_function(ax=ax, ci_show=False)
        
    plt.title("Trimodal Model Survival Curve")
    plt.xlabel("Days Since Diagnosis")
    plt.ylabel("Survival Probability")
    plt.savefig("./km_curve.png")
    # plt.show()