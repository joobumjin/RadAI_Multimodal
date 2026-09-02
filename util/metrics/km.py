import pandas as pd
import torch
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns

def km_preds(all_deaths: torch.Tensor, all_times: torch.Tensor, all_preds: torch.Tensor, model_name: str):
    """
    Builds and plots a KM curve based on predictions
    Specifically splits the prediciton range into 3 equal bins and KM fits to those 3 risk groups

    params:
    - all_deaths: bool tensor indicator for survival event
    - all_times: float tensor for duration from diagnosis to last follow up
    - all_preds: float tensor with model risk scores
    """

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
        
    plt.title(f"{model_name} Survival Curve")
    plt.xlabel("Days Since Diagnosis")
    plt.ylabel("Survival Probability")
    plt.savefig("./km_curve.png")
    # plt.show()