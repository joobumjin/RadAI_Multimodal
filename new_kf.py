import argparse

import numpy as np
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split, KFold
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
    # parser.add_argument('--data_path',          type=str,   default="../{model}_path_rad_text_embs")
    parser.add_argument('--data_path',          type=str,   default="../gemma_multimodal_bins")
    parser.add_argument('--test_path',          type=str,   default="../gemma_multimodal_bins_rw")
    parser.add_argument('--epochs',             type=int,   default=200)
    parser.add_argument('--device',                         default='cuda')
    parser.add_argument('--float16',            type=bool,  default=True)
    parser.add_argument('--early_stop',         type=bool,  default=True)
    parser.add_argument('--patience',           type=int,   default=5)
    parser.add_argument('--folds',              type=int,   default=5)

    parser.add_argument('--mix_data',           action="store_true", 
                        help="Combine and then shuffle the train and test data")
    parser.add_argument('--modality',           type=str, default="multi", choices=["multi", "uni_path", "uni_rad", "uni_clin"])

    parser.add_argument('--sparse',             action="store_true")
    parser.add_argument('--clinical',           action="store_true")
    parser.add_argument('--clinical_imputed',   action="store_true")
    parser.add_argument('--path_lang',          action="store_true")
    parser.add_argument('--rad_lang',           action="store_true")
    parser.add_argument('--path_img',           action="store_true")
    parser.add_argument('--enc_dim',            type=int,   default=768)
    parser.add_argument('--emb_dim',            type=int,   default=256)
    
    parser.add_argument('--prefetch_factor',    type=int,   default=2)
    parser.add_argument('--num_workers',        type=int,   default=1)
    parser.add_argument('--pin_mem',            type=bool,  default=True)
    parser.add_argument('--train_split',        type=float, default=.85)

    parser.add_argument('--label_col',          type=str,   default="survival_days")
    parser.add_argument('--survival_years',     type=int,   default=2)
    parser.add_argument('--censor_col',         type=str,   default="right_censor")

    # Optimizer parameters
    parser.add_argument('--lr',                 type=float, default=1e-3,   metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr',             type=float, default=1e-7,   metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs',      type=int,   default=25,     metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--warmup_start',       type=float, default=1e-2,   metavar='N',
                        help='epochs to warmup LR')
    
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

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if args.modality == "uni_path": 
        args.path_lang = True
        mod_name = "Unimodal Path"
    elif args.modality == "uni_rad": 
        args.rad_lang = True
        mod_name = "Unimodal Rad"
    elif args.modality == "uni_clin": 
        args.clinical = True
        mod_name = "Unimodal Clin"
    else:
        mod_name = []
        for mod, n in zip([args.path_lang, args.rad_lang, args.clinical], ["Path", "Rad", "Clin"]):
            if mod: mod_name.append(n)
        
        if len(mod_name) == 3: mod_name = "Trimodal"
        else: mod_name = "Bimodal " + "+".join(mod_name)

    args.wb_proj = f"{mod_name} {args.model} {args.wb_proj} {args.folds}"

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    if args.mix_data:
        datasets = get_datasets(args)
        dataset = ConcatDataset(datasets) #this is prob a little mem inefficient

        tv, test_inds = train_test_split(
            np.arange(len(dataset)),
            test_size=0.15,
            random_state=args.seed,
        )
    else:
        tv_set, test_set = get_datasets(args)
        tv = np.arange(len(tv_set))

    splits = kf.split(X=tv)

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
        # print(f"Validation Inds: {val_inds}")
        if args.mix_data:
            train_loader, valid_loader, test_loader = get_combined_loaders(args, dataset, train_inds, val_inds, test_inds)
        else:
            train_loader, valid_loader, test_loader = get_subset_loaders(args, tv_set, train_inds, val_inds, test_set)

        run_name = f"Split {split}"

        #NOTE: THE MODEL IS TRAINED AS LOG HAZARD PREDICTING
        run = run_setup(args, model_fns[args.modality], train_loader, valid_loader, test_loader, 
                        run_name = run_name, return_keys=['Test AUC', 'Test C-Index'], ret_model=True, early_stop = None)
                        # early_stop = (190 if mod_name=="Trimodal" else None))
        model, device = run.pop("model")

        for key in run: 
            metrics[key].append(run[key])
            # print(f"{key}: {run[key][-5:]}")

        train = compile_split(model, train_loader, device)
        val = compile_split(model, valid_loader, device)
        test = compile_split(model, test_loader, device)

        all = [torch.cat(l) for l in zip(train, test)]

        # if split != 3:
        ci_res, auc_res = bootstrap_cind_auc(args, test, ci_res, auc_res)
        if split == 4 and mod_name == "Trimodal" and args.mix_data:
            km_preds(*all)

    for key in metrics: metrics = {key: np.vstack(arr) for key, arr in metrics.items()}

    avgs = {key: np.mean(arr, axis=0) for key, arr in metrics.items()}
    errs = {key: stats.sem(arr, axis=0) for key, arr in metrics.items()}

    # avg_t = {key: torch.from_numpy(arr) for key, arr in avgs.items()}
    # top_k = {key: torch.topk(arr, 5) for key, arr in avg_t.items()}

    print(f"\n{mod_name}{', mix data' if args.mix_data else ', inst. sep.'} - {args.survival_years}yr")
    print(f"{"--" * 5}")
    print(f"5-Fold Avg + 95% CI")
    selected = -1
    auc_ci = stats.t.interval(confidence=.95, df=4, loc=avgs['Test AUC'][selected], scale=errs['Test AUC'][selected])
    auc_ci = [float(n) for n in auc_ci]
    c_ci = stats.t.interval(confidence=.95, df=4, loc=avgs['Test C-Index'][selected], scale=errs['Test C-Index'][selected])
    c_ci = [float(n) for n in c_ci]

    auc5fold = f"{avgs['Test AUC'][selected]:.2f} ({auc_ci[0]:.3f}, {auc_ci[1]:.3f})"
    ci5fold = f"{avgs['Test C-Index'][selected]:.2f} ({c_ci[0]:.3f}, {c_ci[1]:.3f})"
    print(f"AUC: {auc5fold}")
    print(f"C Ind: {ci5fold}")


    print(f"{"--" * 5}")
    print(f"Bootstrapped Avg + 95% CI")
    # print(f"AUC: {np.mean(auc_res.bootstrap_distribution):.2f} ({auc_res.confidence_interval.low}, {auc_res.confidence_interval.high})")
    # print(f"C Ind: {np.mean(ci_res.bootstrap_distribution):.2f} ({ci_res.confidence_interval.low}, {ci_res.confidence_interval.high})")
    # print(f"From {len(ci_res.bootstrap_distribution)} bootstrap samples")
    print(str(auc_res))
    print(str(ci_res))
    print(f"From {len(ci_res.entries)} bootstrap samples")

    return mod_name, [auc5fold, ci5fold], [str(auc_res).split(":")[1], str(ci_res).split(":")[1]]
       

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True

    main(args)