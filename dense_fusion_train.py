import argparse

import numpy as np
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split

from data import *
from util import *
from run import run_setup, get_combined_loaders, get_subset_loaders, get_datasets
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
    
    parser.add_argument('--prefetch_factor',    type=int,   default=2)
    parser.add_argument('--num_workers',        type=int,   default=1)
    parser.add_argument('--pin_mem',            type=bool,  default=True)
    parser.add_argument('--train_split',        type=float, default=.85)

    parser.add_argument('--survival_years',     type=int,   default=2)

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

    args, mod_name = get_mod_name(args)
    args.wb_proj = f"{mod_name} {args.model} {args.wb_proj} {args.folds}"

    if args.mix_data:
        datasets = get_datasets(args)
        dataset = ConcatDataset(datasets) #this is prob a little mem inefficient

        all_loader = DataLoader(dataset)
        if args.stratify:
            strat = []
            for s in all_loader:
                time = s['survival_days'] > args.survival_years * DAYS_IN_YR
                event = not s['survival_right_censor']
                strat.append((event * 2 + time)[0,0])
        else:
            strat = None

        tv, test_inds = train_test_split(
            np.arange(len(dataset)),
            test_size=0.15,
            random_state=args.seed,
            stratify=strat
        )

        train_inds, val_inds = tv, []

        train_loader, valid_loader, test_loader = get_combined_loaders(args, dataset, train_inds, val_inds, test_inds)

    else:
        tv_set, test_set = get_datasets(args)
        tv = np.arange(len(tv_set))

        train_inds, val_inds = tv, []

        train_loader, valid_loader, test_loader = get_subset_loaders(args, tv_set, train_inds, val_inds, test_set)

    run_name = args.run_name if args.run_name is not None else " - ".join([f"{args.emb_dim}e{", sparse" if args.sparse else ""}", f"{args.label_col}", f"{args.model} - "])

    model_fns = {
        "multi": get_dense_fusion_model,
        "uni_path": get_unimodal_path,
        "uni_rad": get_unimodal_rad,
        "uni_clin": get_unimodal_clin
    }

    run = run_setup(args, model_fns[args.modality], train_loader, valid_loader, test_loader, run_name = run_name, return_keys=['Test AUC', 'Test C-Index'], ret_model=True, early_stop=990)
    #384
    model, device = run.pop("model")

    train = compile_split(model, train_loader, device)
    val = compile_split(model, valid_loader, device)
    test = compile_split(model, test_loader, device)
    all = [torch.cat(l) for l in zip(train, val, test)]
    if args.plot:
        km_preds(*all)

    finals = {key: arr[-1].item() for key, arr in run.items()}

    # ci_res, auc_res = bootstrap_cind_auc(args, test, None, None)

    # print(f"{"--" * 5}")
    # print(f"Bootstrapped Avg + 95% CI")
    # print(str(auc_res))
    # print(str(ci_res))
    # print(f"From {len(ci_res.entries)} bootstrap samples")
    print(finals)


if __name__ == '__main__':
    parser  = get_args_parser()
    args    = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.disable_wandb = True

    if args.modality == "uni_path": args.path_lang = True
    elif args.modality == "uni_rad": args.rad_lang = True
    elif args.modality == "uni_clin": args.clinical = True

    main(args)