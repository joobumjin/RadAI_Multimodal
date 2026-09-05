import argparse
from types import SimpleNamespace

import numpy as np
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import scipy.stats as stats

from data import *
from util import *
from run import *
from final_models import *

def get_args_parser():
    parser = argparse.ArgumentParser('Supervised Panc Training', add_help=False)

    parser.add_argument('--seed',               type=int,   default=0)
      
    parser.add_argument('--batch_size',         type=int,   default=128)
    parser.add_argument('--loss_fn',            type=str,   default="cox_nll", choices=list(LOSS_FNS.keys()))
    parser.add_argument('--model',              type=str,   default="gemma", choices=['gemma', 'qwen'])
    parser.add_argument('--data_path',          type=str,   default="../gemma_multimodal_bins")
    parser.add_argument('--test_path',          type=str,   default="../gemma_multimodal_bins_rw")
    parser.add_argument('--epochs',             type=int,   default=200)
    parser.add_argument('--device',                         default='cuda')
    parser.add_argument('--float16',            type=bool,  default=True)
    parser.add_argument('--early_stop',         type=bool,  default=True)
    parser.add_argument('--patience',           type=int,   default=5)
    parser.add_argument('--folds',              type=int,   default=5)
    parser.add_argument('--resample',           type=int,   default=9999)

    parser.add_argument('--mix_data',           action="store_true", 
                        help="Combine and then shuffle the train and test data")
    parser.add_argument('--stratify',           action="store_true", help="stratify the mix split")
    parser.add_argument('--modality',           type=str, default="multi", choices=["multi", "uni_path", "uni_rad", "uni_clin"])

    parser.add_argument('--sparse',             action="store_true")
    parser.add_argument('--clinical',           action="store_true")
    parser.add_argument('--clinical_imputed',   action="store_true")
    parser.add_argument('--path_lang',          action="store_true")
    parser.add_argument('--rad_lang',           action="store_true")
    parser.add_argument('--path_img',           action="store_true")
    parser.add_argument('--enc_dim',            type=int,   default=768)
    parser.add_argument('--emb_dim',            type=int,   default=256)
    parser.add_argument('--fusion_dim',         type=int,   default=0)
    parser.add_argument('--rms_norm',           action="store_true")
    parser.add_argument('--layer_norm',         action="store_true")
    parser.add_argument('--batch_norm',         action="store_true")
    
    parser.add_argument('--prefetch_factor',    type=int,   default=2)
    parser.add_argument('--num_workers',        type=int,   default=1)
    parser.add_argument('--pin_mem',            type=bool,  default=True)
    parser.add_argument('--train_split',        type=float, default=.85)

    parser.add_argument('--survival_years',     type=int,   default=2)
    parser.add_argument('--plot',               action="store_true")
    parser.add_argument('--track_best',         action="store_true")

    # Optimizer parameters
    parser.add_argument('--lr',                 type=float, default=1e-4,   metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--const_lr',           action="store_true")
    parser.add_argument('--min_lr',             type=float, default=1e-7,   metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs',      type=int,   default=25,     metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--warmup_start',       type=float, default=1e-2,   metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--early',      type=int,   default=0)
    
    parser.add_argument('--save_path',                      default="../weights",
                        help="directory to which the pretrained model weights should be saved")

    parser.add_argument('--disable_wandb',      action="store_true")
    parser.add_argument('--wb_proj',            type=str,   default="New KFold Validation")
    parser.add_argument('--run_name')
    parser.add_argument('--debug',              action="store_true")
    return parser

# --------------------------------------------------------

def main(args):
    import warnings
    warnings.filterwarnings("ignore", message="Ties in `time` detected")

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True
        args.resample = 5

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    args, mod_name = get_mod_name(args)
    args.wb_proj = f"{mod_name} {args.model} {args.wb_proj} {args.folds}"

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    if args.mix_data:
        datasets = get_datasets(args)
        dataset = ConcatDataset(datasets) #this is prob a little mem inefficient

        if args.stratify:
            all_loader = DataLoader(dataset)
            strat = []
            for s in all_loader:
                time = s['survival_days'] > args.survival_years * DAYS_IN_YR
                event = not s['survival_right_censor']
                strat.append((event * 2 + time)[0,0])
            strat = np.array(strat)
            kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        else: strat = None

        tv, test_inds = train_test_split(
            np.arange(len(dataset)),
            test_size=0.15,
            random_state=args.seed,
            stratify=strat
        )

    else:
        tv_set, test_set = get_datasets(args)
        tv = np.arange(len(tv_set))

    splits = kf.split(X=tv) if not args.stratify else kf.split(X=tv, y=strat[tv])

    model_fns = {
        "multi": get_dense_fusion_model,
        "uni_path": get_unimodal_path,
        "uni_rad": get_unimodal_rad,
        "uni_clin": get_unimodal_clin
    }

    metrics = defaultdict(lambda: [])

    ci_res, auc_res = None, None

    for split, (train_inds, val_inds) in enumerate(splits):
        print(f"\n{"--" * 5}")
        print(f"Split {split}")

        train_inds, val_inds = tv[train_inds], tv[val_inds]
        if args.mix_data:
            train_loader, valid_loader, test_loader = get_combined_loaders(args, dataset, train_inds, val_inds, test_inds)
        else:
            train_loader, valid_loader, test_loader = get_subset_loaders(args, tv_set, train_inds, val_inds, test_set)

        run_name = f"Split {split}"

        run = run_setup(args, model_fns[args.modality], 
                        train_loader, valid_loader, test_loader, 
                        run_name = run_name, return_keys=['Test AUC', 'Test C-Index'], 
                        ret_model=True, early_stop = args.early)
        model, device = run.pop("model")

        for key in run: 
            metrics[key].append(run[key])

        train = compile_split(model, train_loader, device)
        val = compile_split(model, valid_loader, device)
        test = compile_split(model, test_loader, device)

        all = [torch.cat(l) for l in zip(train, val, test)]

        # ci_res, auc_res = bootstrap_cind_auc(args, test, ci_res, auc_res)
        if split == 4 and args.plot:
            km_preds(*all, model_name = f"Trimodal {args.survival_years}yr Model")

    for key in metrics: metrics = {key: np.vstack(arr) for key, arr in metrics.items()}

    avgs = {key: np.mean(arr, axis=0) for key, arr in metrics.items()}
    errs = {key: stats.sem(arr, axis=0) for key, arr in metrics.items()}

    if args.track_best:
        sum = avgs["Test AUC"] + avgs["Test C-Index"]
        best = [np.max(sum[-100:])]
        selected = np.argmax(sum[-100:]) + len(sum) - 100
    else: 
        best = []
        selected = -1

    print(f"\n{mod_name}{', mix data' if args.mix_data else ', inst. sep.'} - {args.survival_years}yr")
    print(f"{"--" * 5}")
    print(f"5-Fold Avg + 95% CI")
    # for selected in [654, 816, 398]:
    # selected = -1
    auc_ci = stats.t.interval(confidence=.95, df=4, loc=avgs['Test AUC'][selected], scale=errs['Test AUC'][selected])
    auc_ci = [float(n) for n in auc_ci]
    c_ci = stats.t.interval(confidence=.95, df=4, loc=avgs['Test C-Index'][selected], scale=errs['Test C-Index'][selected])
    c_ci = [float(n) for n in c_ci]
    print(f"At epoch {selected}")

    auc5fold = SimpleNamespace(avg = avgs['Test AUC'][selected], ci=auc_ci)
    ci5fold = SimpleNamespace(avg = avgs['Test C-Index'][selected], ci=c_ci)
    print(f"AUC: {auc5fold.avg} ({auc_ci[0]:.3f}, {auc_ci[1]:.3f})")
    print(f"C Ind: {ci5fold.avg} ({c_ci[0]:.3f}, {c_ci[1]:.3f})")

    # print(f"{"--" * 5}")
    # print(f"Bootstrapped Avg + 95% CI")
    # print(str(auc_res))
    # print(str(ci_res))
    # print(f"From {len(ci_res.entries)} bootstrap samples")

    return mod_name, [auc5fold, ci5fold], [auc_res, ci_res], *best
       

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    main(args)