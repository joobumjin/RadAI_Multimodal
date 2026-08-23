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
      
    parser.add_argument('--batch_size',         type=int,   default=16)
    parser.add_argument('--loss_fn',            type=str,   default="bce", choices=list(LOSS_FNS.keys()))
    parser.add_argument('--model',              type=str,   default="conch", choices=['conch', 'biomedclip', 'gemma', 'qwen'])
    # parser.add_argument('--data_path',          type=str,   default="../{model}_path_rad_text_embs")
    parser.add_argument('--data_path',          type=str,   default="../updated_multimodal_bins")
    parser.add_argument('--test_path',          type=str,   default="../multimodal_bins_rw")
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
    parser.add_argument('--enc_dim',            type=int,   default=512)
    parser.add_argument('--emb_dim',            type=int,   default=256)
    
    parser.add_argument('--prefetch_factor',    type=int,   default=2)
    parser.add_argument('--num_workers',        type=int,   default=1)
    parser.add_argument('--pin_mem',            type=bool,  default=True)
    parser.add_argument('--train_split',        type=float, default=.85)

    parser.add_argument('--label_col',          type=str,   default="survival_days")
    parser.add_argument('--survival_years',     type=int,   default=2)
    parser.add_argument('--censor_col',         type=str,   default="right_censor")

    # Optimizer parameters
    parser.add_argument('--lr',                 type=float, default=1e-4,   metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr',             type=float, default=1e-7,   metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs',      type=int,   default=50,     metavar='N',
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

def get_inds(args):
    keys = ["slide_ids", "survival_days", "survival_right_censor"]

    index   = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)
    inds    = np.arange(len(index['survival_days']))

    label_mask = ~np.isnan(index[args.label_col])
    exclusion_mask = ~index["excluded"]
    mask = label_mask & exclusion_mask
    if "indicator" in args.label_col: 
        for key in keys:
            mask = mask & (~np.isnan(index[key]))

    modality_mask = np.zeros_like(mask).astype(bool) if args.sparse else np.ones_like(mask).astype(bool)
    combine_op = lambda x, y: x | y if args.sparse else x & y

    arg_dict = vars(args)
    for mod in ["clinical", "clinical_imputed"]:
        if arg_dict.get(mod, False): 
            modality_mask = combine_op(modality_mask, index[f'{mod}_mask'])
    for mod in ["path_lang", "rad_lang", "path_img"]:
        if arg_dict.get(mod, False): 
            modality_mask = combine_op(modality_mask, (index[f'{mod}_mask']))

    mask = mask & modality_mask

    valid_inds = inds[mask]

    return valid_inds

# --------------------------------------------------------

def all_compile(model, device, train_loader, valid_loader, test_loader):
    model.eval()
    train_preds, train_deaths, train_times = [], [], []
    test_preds, test_deaths, test_times = [], [], []
    # for loader, testing in zip([train_loader, valid_loader, test_loader], [False, False, True]):
    for loader, (preds, deaths, times) in zip([train_loader, test_loader], [(train_preds, train_deaths, train_times), (test_preds, test_deaths, test_times)]):
        for batch in loader:
            surviving = (batch["survival_right_censor"].squeeze(-1)) #true if could have lived longer
            times = batch["survival_days"].squeeze(-1)
            deaths.append(~surviving)
            times.append(times)

            for key in batch: batch[key] = batch[key].to(device)
            with torch.inference_mode():
                preds = model.predict(batch).detach().squeeze(-1).cpu() #predicted hazards
                preds.append(preds)
    
    train = [torch.cat(l) for l in [train_deaths, train_times, train_preds]]
    test = [torch.cat(l) for l in [test_deaths, test_times, test_preds]]

    all = [torch.cat(l) for l in [(train_deaths, test_deaths), (train_times, test_times), (train_preds, test_preds)]]
    return all, train, test

def km_preds(all_deaths, all_times, all_preds):
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

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
    plt.show()


def bootstrap_cind_auc(args, train_samples, test_samples, ci_res, auc_res):
    surv_train = np.zeros(len(train_samples[0]), dtype=[('event', bool), ('surv_time', float)])
    surv_train['event'] = train_samples[0]
    surv_train['surv_time'] = train_samples[1]

    def cind(death, times, preds):
        death = death.astype(bool)
        c = concordance_index_censored(death, times, preds)[0]

        return c

    def auc(death, times, preds):
        death = death.astype(bool)
        surv_test = np.zeros(len(death), dtype=[('event', bool), ('surv_time', float)])
        surv_test['event'] = death
        surv_test['surv_time'] = times
        auc = cumulative_dynamic_auc(surv_train, surv_test, preds, [args.survival_years * DAYS_IN_YR], tied_tol=1e-08)[0][0]
        return auc
    
    # ci_res = stats.bootstrap(test_samples, cind, vectorized=False, paired=True, bootstrap_result=ci_res, n_resamples=175, method='percentile')
    # auc_res = stats.bootstrap(test_samples, auc, vectorized=False, paired=True, bootstrap_result=auc_res, n_resamples=175, method='percentile')
    
    if ci_res is None:
        ci_res, auc_res = Bootstrap("C-Index", cind, n_resample=100), Bootstrap("AUC", auc, n_resample=100)

    ci_res.add(*test_samples)
    auc_res.add(*test_samples)

    return ci_res, auc_res


def main(args):
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
        run = run_setup(args, model_fns[args.modality], train_loader, valid_loader, test_loader, run_name = run_name, return_keys=['Test AUC', 'Test C-Index'], ret_model=True)
        model, device = run.pop("model")

        for key in run: 
            metrics[key].append(run[key])
            print(f"{key}: {run[key][-5:]}")

        all, train, test = all_compile(model, device, train_loader, valid_loader, test_loader)
        # if split != 3:
        ci_res, auc_res = bootstrap_cind_auc(args, train, test, ci_res, auc_res)
        if split == 4 and mod_name == "Trimodal":
            km_preds(*all)

    for key in metrics: metrics = {key: np.vstack(arr) for key, arr in metrics.items()}

    avgs = {key: np.mean(arr, axis=0) for key, arr in metrics.items()}
    errs = {key: stats.sem(arr, axis=0) for key, arr in metrics.items()}

    # avg_t = {key: torch.from_numpy(arr) for key, arr in avgs.items()}
    # top_k = {key: torch.topk(arr, 5) for key, arr in avg_t.items()}

    print(f"\n{mod_name}")
    print(f"{"--" * 5}")
    print(f"5-Fold Avg + 95% CI")
    selected = -1
    auc_ci = stats.t.interval(confidence=.95, df=4, loc=avgs['Test AUC'][selected], scale=errs['Test AUC'][selected])
    auc_ci = [float(n) for n in auc_ci]
    c_ci = stats.t.interval(confidence=.95, df=4, loc=avgs['Test C-Index'][selected], scale=errs['Test C-Index'][selected])
    c_ci = [float(n) for n in c_ci]

    print(f"AUC: {avgs['Test AUC'][selected]:.2f} ({auc_ci[0]:.2f}, {auc_ci[1]:.2f})")
    print(f"C Ind: {avgs['Test C-Index'][selected]:.2f} ({c_ci[0]:.2f}, {c_ci[1]:.2f})")


    print(f"{"--" * 5}")
    print(f"Bootstrapped Avg + 95% CI")
    # print(f"AUC: {np.mean(auc_res.bootstrap_distribution):.2f} ({auc_res.confidence_interval.low:.2f}, {auc_res.confidence_interval.high:.2f})")
    # print(f"C Ind: {np.mean(ci_res.bootstrap_distribution):.2f} ({ci_res.confidence_interval.low:.2f}, {ci_res.confidence_interval.high:.2f})")
    print(str(auc_res))
    print(str(ci_res))
       

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True

    import warnings
    warnings.filterwarnings("ignore", message="Ties in `time` detected")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    main(args)