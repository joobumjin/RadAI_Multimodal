import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from lifelines import KaplanMeierFitter
import scipy.stats as stats

from data import *
from run import *
from util import bootstrap_cind_auc, cind, auc


def get_args_parser():
    parser = argparse.ArgumentParser('Supervised Panc Training', add_help=False)

    parser.add_argument('--seed',               type=int,   default=0)
      
    parser.add_argument('--data_path',          type=str,   default="../gemma_multimodal_bins")
    parser.add_argument('--test_path',          type=str,   default="../gemma_multimodal_bins_rw")
    parser.add_argument('--folds',              type=int,   default=5)
    parser.add_argument('--resample',           type=int,   default=9999)

    parser.add_argument('--mix_data',           action="store_true", 
                        help="Combine and then shuffle the train and test data")
    parser.add_argument('--stratify',           action="store_true", help="stratify the mix split")
    
    parser.add_argument('--test_split',         type=float, default=.15)
    parser.add_argument('--sparse',             action="store_true")

    parser.add_argument('--survival_years',     type=int,   default=2)
    parser.add_argument('--debug',              action="store_true")
    return parser

def build_data(args):
    train_set, test_set = get_datasets(args, keys=["slide_ids", "survival_days", "survival_right_censor", "stage"]) 

    if args.mix_data:
        all_ds = ConcatDataset([train_set, test_set])
        all_loader = DataLoader(all_ds)
        stages, time, event = [], [], []
        strat = [] if args.stratify else None
        for s in all_loader:
            stages.append(s['stage'].item())
            time.append(s['survival_days'].item())
            event.append(not s['survival_right_censor'].item())
            if args.stratify: strat.append((not s['survival_right_censor'] * 2 + s['survival_days'] > args.survival_years * DAYS_IN_YR)[0,0])
        if args.stratify: strat = np.array(strat)

        df = pd.DataFrame({"Stage": stages, "Survival Time": time, "Survival Event": event})

        tv_inds, test_inds = train_test_split(
            np.arange(len(df)),
            test_size=args.test_split,
            random_state=args.seed,
            stratify=strat
        )
        return df, df, tv_inds, test_inds, strat

    train_loader, test_loader = DataLoader(train_set), DataLoader(test_set)

    train_stages, train_time, train_event = [], [], []
    for s in train_loader:
        train_stages.append(s['stage'].item())
        train_time.append(s['survival_days'].item())
        train_event.append(not s['survival_right_censor'].item())
    tv_df = pd.DataFrame({"Stage": train_stages, "Survival Time": train_time, "Survival Event": train_event})
    tv_inds = np.arange(len(tv_df))

    test_stages, test_time, test_event = [], [], []
    for s in test_loader:
        test_stages.append(s['stage'].item())
        test_time.append(s['survival_days'].item())
        test_event.append(not s['survival_right_censor'].item())
    test_df = pd.DataFrame({"Stage": test_stages, "Survival Time": test_time, "Survival Event": test_event})
    test_inds = np.arange(len(test_df))

    return tv_df, test_df, tv_inds, test_inds, None


def main(args):
    if args.debug:
        args.resample = 5

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    tv_df, test_df, tv_inds, test_inds, strat = build_data(args)

    test = test_df.iloc[test_inds]

    if not args.stratify:
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed) 
        splits = kf.split(X=tv_inds) 
    else: 
        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = kf.split(X=tv_inds, y=strat[tv_inds])

    ci_res, auc_res = None, None

    for split, (train_inds, val_inds) in enumerate(splits):
        print(f"Split {split}")
        train_inds, val_inds = tv_inds[train_inds], tv_inds[val_inds]
        train, val = tv_df.iloc[train_inds], tv_df.iloc[val_inds]
        kmf = KaplanMeierFitter()
        s_prop = {}

        for name, grouped_df in train.groupby("Stage"):
            kmf.fit(
                grouped_df["Survival Time"],
                event_observed=grouped_df["Survival Event"],
                label=name,
            )
            s_prop[name] = dict(kmf.survival_function_at_times([args.survival_years * DAYS_IN_YR]))

        km_risk = 1 - pd.DataFrame.from_dict(s_prop, orient='index')

        df_km_2yr = test['Stage'].map(dict(km_risk[args.survival_years * DAYS_IN_YR]))

        test_samples = [torch.from_numpy(test['Survival Event'].to_numpy(dtype=bool).copy()), 
                        torch.from_numpy(test['Survival Time'].to_numpy().copy()),
                        torch.from_numpy(df_km_2yr.to_numpy().copy())]
        ci_res, auc_res = bootstrap_cind_auc(args, test_samples, ci_res, auc_res)

    print(f"{"--" * 5}")
    print(f"Bootstrapped Avg + 95% CI")
    print(str(auc_res))
    print(str(ci_res))
    print(f"From {len(ci_res.entries)} bootstrap samples")

    return "KM Stage", [None, None], [auc_res, ci_res]


if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    main(args)