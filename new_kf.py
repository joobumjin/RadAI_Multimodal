import argparse

import numpy as np
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split, KFold
import scipy.stats as stats

from data import *
from util import *
from run import *
from dense_fusion_train import get_dense_fusion_model, get_unimodal_clin, get_unimodal_path, get_unimodal_rad

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
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    datasets = get_datasets(args)
    dataset = ConcatDataset(datasets) #this is prob a little mem inefficient

    tv, test_inds = train_test_split(
        np.arange(len(dataset)),
        test_size=0.15,
        random_state=args.seed,
        #consider stratifying
    )


    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = kf.split(X=tv)

    model_fns = {
        "multi": get_dense_fusion_model,
        "uni_path": get_unimodal_path,
        "uni_rad": get_unimodal_rad,
        "uni_clin": get_unimodal_clin
    }

    metrics = defaultdict(lambda: [])

    for split, (train_inds, val_inds) in enumerate(splits):
        train_loader, valid_loader, test_loader = get_combined_loaders(args, dataset, train_inds, val_inds, test_inds)

        run_name = f"Split {split}"

        #NOTE: THE MODEL IS TRAINED AS LOG HAZARD PREDICTING
        run = run_setup(args, model_fns[args.modality], train_loader, valid_loader, test_loader, run_name = run_name, return_keys=['Test AUC', 'Test C-Index'], ret_model=(split==4))
        if split == 4:
            model, device  = run.pop("model")
        for key in run: metrics[key].append(run[key])

    for key in metrics: metrics = {key: np.vstack(arr) for key, arr in metrics.items()}

    avgs = {key: np.mean(arr, axis=0) for key, arr in metrics.items()}
    errs = {key: stats.sem(arr, axis=0) for key, arr in metrics.items()}

    avg_t = {key: torch.from_numpy(arr) for key, arr in avgs.items()}
    top_k = {key: torch.topk(arr, 5) for key, arr in avg_t.items()}

    print(mod_name)
    print(f"Average Scores Across Splits: {avgs}")
    print(f"Standard Error Across Splits: {errs}")
    print(f"Top-k Average Inds: {top_k}")


    model.eval()

    all_preds, all_deaths, all_times = [], [], []
    # for loader in train_loader, valid_loader, test_loader:
    for loader in train_loader, test_loader:
        for batch in loader:
            surviving = (batch["survival_right_censor"].numpy().squeeze(-1).astype(bool)) #true if could have lived longer
            times = batch["survival_days"].numpy().squeeze(-1)
            all_deaths.append(~surviving)
            all_times.append(times)

            for key in batch: batch[key] = batch[key].to(device)
            with torch.inference_mode():
                preds = model.predict(batch).detach().squeeze(-1).cpu().numpy() #predicted hazards
                all_preds.append(preds)
    
    all_deaths, all_times, all_preds = [np.concatenate(l) for l in [all_deaths, all_times, all_preds]]

    pred_bins = np.linspace(np.min(all_preds), np.max(all_preds), 4)
    pred_binned = np.digitize(all_preds, pred_bins)

    df = pd.DataFrame({"Death": all_deaths, "Time": all_times, "Pred Bins": pred_binned})

    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

    # Initialize the fitter
    kmf = KaplanMeierFitter()

    ax = plt.subplot(111)

    # Fit and plot each group separately onto the same axis (ax)
    #group by stratification and plot the time and event
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
       

if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True

    import warnings
    warnings.filterwarnings("ignore", message="Ties in `time` detected")

    main(args)