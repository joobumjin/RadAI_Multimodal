import argparse

import numpy as np
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import *
from util import *
from run import run_setup, get_combined_loaders, get_subset_loaders, get_datasets
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
    
    #optimizer
    parser.add_argument('--epochs',             type=int,   default=500)
    parser.add_argument('--early_stop',         type=bool,  default=True)
    parser.add_argument('--patience',           type=int,   default=5)
    parser.add_argument('--lr',                 type=float, default=1e-3,       help='learning rate (absolute lr)')
    parser.add_argument('--min_lr',             type=float, default=1e-7,       help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--const_lr',           action="store_true")
    parser.add_argument('--warmup_epochs',      type=int,   default=25,         help='epochs to warmup LR')
    parser.add_argument('--warmup_start',       type=float, default=1e-2,       help='epochs to warmup LR')
    parser.add_argument('--early',              type=int,   default=0)

    #run infra
    parser.add_argument('--hist',               action="store_true")
    parser.add_argument('--plot',               action="store_true")
    parser.add_argument('--resample',           type=int,   default=999)
    parser.add_argument('--track_best',         action="store_true")
    parser.add_argument('--disable_wandb',      action="store_true")
    parser.add_argument('--wb_proj',            type=str,   default="5CV")
    parser.add_argument('--run_name')
    parser.add_argument('--debug',              action="store_true")
    return parser

# --------------------------------------------------------

def get_stratify(args: Namespace, dataset: Dataset):
    strat = []
    for s in DataLoader(dataset):
        time = s['survival_days'] > args.survival_years * DAYS_IN_YR
        event = not s['survival_right_censor']
        strat.append((event * 2 + time)[0,0])
    strat = np.array(strat)
    return strat

def main(args):
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

    # train_inds, val_inds = train_test_split(tv_inds, test_size=.08, random_state=args.seed, stratify=strat[tv_inds])
    train_inds, val_inds = tv_inds, []

    if args.mix_data:
        train_loader, valid_loader, test_loader = get_combined_loaders(args, dataset, train_inds, val_inds, test_inds)
    else:
        train_loader, valid_loader, test_loader = get_subset_loaders(args, dataset, train_inds, val_inds, test_set)

    run_name = args.run_name if args.run_name is not None else " - ".join([f"{args.emb_dim}e{", sparse" if args.sparse else ""}", f"{args.label_col}", f"{args.model} - "])

    run = run_setup(args, MODEL_FNS[args.modality], 
                        train_loader, valid_loader, test_loader, 
                        run_name = run_name, return_keys=['Train Loss', 'Train AUC', 'Train C-Index', 'Test Loss', 'Test AUC', 'Test C-Index'], 
                        ret_model=True, early_stop = args.early)
    model, device = run.pop("model")

    if args.plot:
        train = compile_split(model, train_loader, device)
        val = compile_split(model, valid_loader, device)
        test = compile_split(model, test_loader, device)

        all = [torch.cat(l) for l in zip(train, val, test)]
        km_preds(*all, model_name = f"Trimodal", setup=f"{args.survival_years}yr {"mixed" if args.mix_data else "iso"}")
        del all

        inp_weights, pair_weights = compile_attn(model, test_loader, device)
        vis_hat(inp_weights, pair_weights, model_name = f"Trimodal {args.survival_years}yr")
        del inp_weights, pair_weights

        # logits, labels = compile_logits(model, device, [train_loader, valid_loader, test_loader], args)
        # plot_umap(logits["path_lang"], labels, f"Path Lang, Trimodal {args.survival_years}yr Split{split}")
        # plot_umap(logits["rad_lang"], labels, f"Rad Lang, Trimodal {args.survival_years}yr Split{split}")
        # del logits, labels

    #explicitly dealloc
    del model, train_loader, valid_loader, test_loader

    if args.hist:

        epochs = np.arange(len(list(metrics.values())[0][0]))

        fig = make_subplots(rows=2, cols=3, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=list(run.keys()))
        for i, key in enumerate(run):
            fig.add_trace(go.Scatter(x=epochs, y=run[key], mode='lines', name=key), row=i//3 + 1, col=(i%3) + 1)

        fig.show()

    finals = {key: arr[-1].item() for key, arr in run.items()}

    ci_res, auc_res = bootstrap_cind_auc(args, test, None, None)

    print(f"{"--" * 5}")
    print(f"Bootstrapped Avg + 95% CI")
    print(str(auc_res))
    print(str(ci_res))
    print(f"From {len(ci_res.entries)} bootstrap samples")
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