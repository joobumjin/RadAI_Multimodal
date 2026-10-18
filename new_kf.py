import argparse
from types import SimpleNamespace

import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import scipy.stats as stats

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import *
from util import *
from run import *
from final_models import *

def get_args_parser():
    parser = argparse.ArgumentParser('Supervised Panc Training', add_help=False)

    parser.add_argument('--seed',               type=int,   default=0)

    #dataset  
    parser.add_argument('--data_path',          type=str,   default="../data/gemma_multimodal_bins")
    parser.add_argument('--test_path',          type=str,   default="../data/gemma_multimodal_bins_rw")

    #data sampling
    parser.add_argument('--mix_data',           action="store_true", 
                        help="Combine and then shuffle the train and test data")
    parser.add_argument('--stratify',           action="store_true", help="stratify the mix split")
    parser.add_argument('--modality',           type=str, default="multi", choices=list(MODEL_FNS.keys()))

    #modalities
    parser.add_argument('--sparse',             action="store_true")
    parser.add_argument('--clinical',           action="store_true")
    parser.add_argument('--clinical_imputed',   action="store_true")
    parser.add_argument('--path_lang',          action="store_true")
    parser.add_argument('--rad_lang',           action="store_true")
    parser.add_argument('--path_img',           action="store_true")

    #model_setup
    parser.add_argument('--loss_fn',            type=str,   default="cox_nll", choices=list(LOSS_FNS.keys()))
    parser.add_argument('--enc_dim',            type=int,   default=768)
    parser.add_argument('--emb_dim',            type=int,   default=256)
    parser.add_argument('--fusion_dim',         type=int,   nargs='+',      default=[64, 16])
    parser.add_argument('--rms_norm',           action="store_true")
    parser.add_argument('--layer_norm',         action="store_true")
    parser.add_argument('--batch_norm',         action="store_true")

    #dataloader
    parser.add_argument('--batch_size',         type=int,   default=128)
    parser.add_argument('--survival_years',     type=int,   default=2)
    parser.add_argument('--prefetch_factor',    type=int,   default=2)
    parser.add_argument('--num_workers',        type=int,   default=1)
    parser.add_argument('--pin_mem',            type=bool,  default=False)
    parser.add_argument('--train_split',        type=float, default=.85)
    
    #optimizer
    parser.add_argument('--epochs',             type=int,   default=500)
    parser.add_argument('--early_stop',         type=bool,  default=True)
    parser.add_argument('--patience',           type=int,   default=5)
    parser.add_argument('--folds',              type=int,   default=5)
    parser.add_argument('--lr',                 type=float, default=1e-3,       help='learning rate (absolute lr)')
    parser.add_argument('--min_lr',             type=float, default=1e-7,       help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--const_lr',           action="store_true")
    parser.add_argument('--warmup_epochs',      type=int,   default=25,         help='epochs to warmup LR')
    parser.add_argument('--warmup_start',       type=float, default=1e-2,       help='epochs to warmup LR')
    parser.add_argument('--early',              type=int,   default=0)

    #run infra
    parser.add_argument('--hist',               action="store_true")
    parser.add_argument('--plot',               action="store_true")
    parser.add_argument('--track_best',         action="store_true")
    parser.add_argument('--disable_wandb',      action="store_true")
    parser.add_argument('--return_extras',      action="store_true")
    parser.add_argument('--wb_proj',            type=str,   default="5CV")
    parser.add_argument('--run_name')
    parser.add_argument('--debug',              action="store_true")
    return parser

# --------------------------------------------------------

def get_stratify(args: Namespace, dataset: Dataset) -> np.ndarray:
    """
    Returns stratification labels for every sample in the dataset
    
    Parameters
    ----------
    args : argparser.Namespace
        the parsed cli arguments
    dataset : Dataset
        the dataset with n samples to be stratification labeled 
    
    Returns
    ----------
    strat : np.ndarray 
        Array of shape (n, ) with values from 0 to 3
        - 0: Censored, Died in under args.survival_years
        - 1: Censored, Did not die in under args.survival_years
        - 2: Uncensored, Died in under args.survival_years
        - 3: Uncensored, Did not die in under args.survival_years
    """
    strat = []
    for s in DataLoader(dataset):
        time = s['survival_days'] > (args.survival_years * DAYS_IN_YR)
        event = not s['survival_right_censor']
        strat.append((event * 2 + time)[0,0])
    strat = np.array(strat)
    return strat

def main(args: Namespace) -> tuple[str, list[SimpleNamespace], list[SimpleNamespace], Optional[list[dict]], Optional[list]]:
    """
    Runs KFold Cross Validation setup
    
    Parameters
    ----------
    args :  Namespace
        the parsed cli arguments

    Returns
    ----------
    mod_name : str
        the name of the modality setup
    5fold_res : list[SimpleNamespace]
        the AUC and C-Index testing results under 5Fold CV
    boot_res : list[SimpleNamespace]
        the AUC and C-Index testing results under bootstrapping in 5fold CV
    extras : Optional[list[dict]]
        a list of dictionaries with any extra details about the setup, if args.return_extras is true
    best : Optional[list]
        the best merged score from training, if args.track_best is true
    """

    import warnings
    warnings.filterwarnings("ignore", message="Ties in `time` detected")

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True
        args.resample = 5
        torch.autograd.set_detect_anomaly(True)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    args, mod_name = get_mod_name(args)
    args.wb_proj = f"{mod_name} {args.wb_proj} {args.folds}"

    dataset, test_set = get_datasets(args)
    if args.mix_data:
        dtypes = dataset._dtypes 
        dataset = ConcatDataset([dataset, test_set]) #this is prob a little mem inefficient
        dataset._dtypes = dtypes

    strat = get_stratify(args, dataset) if args.stratify else None

    if args.mix_data:
        tv_inds, test_inds = train_test_split(np.arange(len(dataset)), test_size=0.15, 
                                              random_state=args.seed, stratify=strat)
    else:
        tv_inds, test_inds = np.arange(len(dataset)), np.arange(len(test_set))

    if args.stratify:
        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)  
    else:
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = kf.split(X=tv_inds) if not args.stratify else kf.split(X=tv_inds, y=strat[tv_inds])

    metrics = defaultdict(lambda: [])

    ci_res, auc_res = None, None

    for split, (train_inds, val_inds) in enumerate(splits):
        print(f"\n{"--" * 5}")
        print(f"Split {split}")

        train_inds, val_inds = tv_inds[train_inds], tv_inds[val_inds]
        if args.mix_data:
            train_loader, valid_loader, test_loader = get_combined_loaders(args, dataset, train_inds, val_inds, test_inds)
        else:
            train_loader, valid_loader, test_loader = get_subset_loaders(args, dataset, train_inds, val_inds, test_set)

        run_name = f"Split {split}"

        run = run_setup(args, MODEL_FNS[args.modality], 
                        train_loader, valid_loader, test_loader, 
                        run_name = run_name, return_keys=['Train Loss', 'Train AUC', 'Train C-Index', 'Test Loss', 'Test AUC', 'Test C-Index'], 
                        ret_model=True)
        model, device = run.pop("model")

        for key in run: 
            metrics[key].append(run[key])

        # ci_res, auc_res = bootstrap_cind_auc(args, test, ci_res, auc_res)
        if args.plot:
            train = compile_split(model, train_loader, device)
            val = compile_split(model, valid_loader, device)
            test = compile_split(model, test_loader, device)
    
            all = [torch.cat(l) for l in zip(train, val, test)]
            km_preds(*all, model_name = f"Trimodal", setup=f"{args.survival_years}yr Split{split} {"mixed" if args.mix_data else "iso"}")
            del all

            inp_weights, pair_weights = compile_attn(model, test_loader, device)
            vis_hat(inp_weights, pair_weights, model_name = f"Trimodal {args.survival_years}yr Split{split}")
            del inp_weights, pair_weights

            # logits, labels = compile_logits(model, device, [train_loader, valid_loader, test_loader], args)
            # plot_umap(logits["path_lang"], labels, f"Path Lang, Trimodal {args.survival_years}yr Split{split}")
            # plot_umap(logits["rad_lang"], labels, f"Rad Lang, Trimodal {args.survival_years}yr Split{split}")
            # del logits, labels

        #explicitly dealloc
        del model, train_loader, valid_loader, test_loader

    metrics = {key: np.vstack(arr) for key, arr in metrics.items()}

    avgs = {key: np.mean(arr, axis=0) for key, arr in metrics.items()}
    errs = {key: stats.sem(arr, axis=0) for key, arr in metrics.items()}

    if args.hist:
        tracked = {
            key: {
                f"Split {i}": hist for i, hist in enumerate(arrs)
            } for key, arrs in metrics.items()
        }

        epochs = np.arange(len(list(metrics.values())[0][0]))

        fig = make_subplots(rows=2, cols=3, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=list(tracked.keys()))
        for i, key in enumerate(tracked):
            fig.add_trace(go.Scatter(x=epochs, y=avgs[key], mode='lines', name=key), row=i//3 + 1, col=(i%3) + 1)
            fig.add_trace(
                go.Scatter(x=np.hstack((epochs, epochs[::-1])), 
                           y=np.hstack((np.max(metrics[key], axis=0), np.min(metrics[key], axis=0)[::-1])), 
                           mode='lines', name=split, fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), 
                           hoverinfo="skip", showlegend=False,), 
                row=i//3 + 1, col=(i%3) + 1)

        fig.show()

    if args.track_best:
        sum = avgs["Test AUC"] / .9 + avgs["Test C-Index"] / .7
        last = - args.epochs // 5
        best = [np.max(sum[last:])]
        selected = np.argmax(sum[last:]) + len(sum) + last
    else: 
        best = []
        selected = -1

    print(f"\n{mod_name}{', mix data' if args.mix_data else ', inst. sep.'} - {args.survival_years}yr")
    print(f"{"--" * 5}")
    print(f"5-Fold Avg + 95% CI at epoch {selected}")

    auc_ci = stats.t.interval(confidence=.95, df=4, loc=avgs['Test AUC'][selected], scale=errs['Test AUC'][selected])
    auc_ci = [float(n) for n in auc_ci]
    c_ci = stats.t.interval(confidence=.95, df=4, loc=avgs['Test C-Index'][selected], scale=errs['Test C-Index'][selected])
    c_ci = [float(n) for n in c_ci]

    auc5fold = SimpleNamespace(avg = avgs['Test AUC'][selected], ci=auc_ci)
    ci5fold = SimpleNamespace(avg = avgs['Test C-Index'][selected], ci=c_ci)

    train_auc_ci = stats.t.interval(confidence=.95, df=4, loc=avgs['Train AUC'][selected], scale=errs['Train AUC'][selected])
    train_auc_ci = [float(n) for n in train_auc_ci]
    train_c_ci = stats.t.interval(confidence=.95, df=4, loc=avgs['Train C-Index'][selected], scale=errs['Train C-Index'][selected])
    train_c_ci = [float(n) for n in train_c_ci]
    train_auc5fold = SimpleNamespace(avg = avgs['Train AUC'][selected], ci=train_auc_ci)
    train_ci5fold = SimpleNamespace(avg = avgs['Train C-Index'][selected], ci=train_c_ci)

    print(f"Train AUC: \t{train_auc5fold.avg:.3f} ({train_auc5fold.ci[0]:.3f}, {train_auc5fold.ci[1]:.3f})"
          f"\t C-Index: {train_ci5fold.avg:.3f} ({train_ci5fold.ci[0]:.3f}, {train_ci5fold.ci[1]:.3f})")
    print(f"Test AUC: \t{auc5fold.avg:.3f} ({auc5fold.ci[0]:.3f}, {auc5fold.ci[1]:.3f})" 
          f"\t C-Index: {ci5fold.avg:.3f} ({ci5fold.ci[0]:.3f}, {ci5fold.ci[1]:.3f})")

    extras = []
    if args.return_extras: 
        extras.append({
            "Train AUC": train_auc5fold.avg,
            "Train C-Index": train_ci5fold.avg
        })

    # print(f"{"--" * 5}")
    # print(f"Bootstrapped Avg + 95% CI (at end)")
    # print(str(auc_res))
    # print(str(ci_res))
    # print(f"From {len(ci_res.entries)} bootstrap samples")

    return mod_name, [auc5fold, ci5fold], [auc_res, ci_res], *extras, *best
       

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    main(args)