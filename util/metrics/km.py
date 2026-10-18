import pandas as pd
import numpy as np
import torch
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
import seaborn as sns

from ..consts import DAYS_IN_MONTH

def km_preds(all_deaths: torch.Tensor, all_times: torch.Tensor, all_preds: torch.Tensor, model_name: str, setup: str, num_groups: int = 3):
    """
    Builds and plots a KM curve based on predictions
    Specifically splits the prediciton range into 3 equal bins and KM fits to those 3 risk groups

    Parameters
    ----------
    all_deaths : torch.Tensor
        bool tensor indicator for survival event
    all_times : torch.Tensor
        float tensor for duration from diagnosis to last follow up
    all_preds : torch.Tensor
        float tensor with model risk scores
    model_name : str
        name of model for plot
    setup : str
        name of experimental setup for plot
    num_groups : int
        number of groups to bin patients into
    """
    
    sns.set_theme()
    sns.set_style({'axes.grid' : False})
    sns.set_palette("deep")

    pred_bins = torch.logspace(torch.min(all_preds), torch.max(all_preds), num_groups)
    pred_binned = torch.bucketize(all_preds, pred_bins)

    df = pd.DataFrame({"Death": all_deaths, "Time": all_times, "Pred Bins": pred_binned})
    df["Time"] = df["Time"] / DAYS_IN_MONTH
    df.loc[df['Time'] > 60.01, 'Death'] = False
    df['Time'] = df['Time'].clip(upper=60.01)

    # Initialize the fitter
    ax = plt.subplot(111)
    kmfs = []

    for name, grouped_df in df.groupby("Pred Bins"):
        kmf = KaplanMeierFitter()
        kmf.fit(grouped_df["Time"], event_observed=grouped_df["Death"], label=f"Group {name}")
        kmf.plot_survival_function(ax=ax, ci_show=False)
        
        kmfs.append(kmf)

    add_at_risk_counts(*kmfs, ax=ax, xticks=np.arange(0, 70, 10), rows_to_show=["At risk"])
    ax.grid(axis='x', linestyle='-')
    ax.grid(axis='y', linestyle='-')
    ax.set_axisbelow(True)

    plt.title(f"{model_name} Hazard Prediction-Derived KM Survival Curve")
    ax.set_xlabel("Months Since Diagnosis")
    ax.xaxis.set_label_position("bottom")
    ax.set_ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(f"vises/km/{model_name}_{setup}_km_curve.pdf")
    plt.close()