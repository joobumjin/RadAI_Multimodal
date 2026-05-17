from argparse import Namespace

import numpy as np
from tabulate import tabulate

from data import *

def get_loaders(args: Namespace, train_inds, validation_inds):
    """
    Required Args: 
    parser.add_argument('--data_path',          type=str,   default="../updated_multimodal_bins")

    parser.add_argument('--label_col',          type=str,   default="survival_days")

    parser.add_argument('--sparse',             action="store_true")
    parser.add_argument('--clinical',           action="store_true")
    parser.add_argument('--clinical_imputed',   action="store_true")
    parser.add_argument('--path_lang',          action="store_true")
    parser.add_argument('--rad_lang',           action="store_true")
    parser.add_argument('--path_img',           action="store_true")

    parser.add_argument('--batch_size',         type=int,   default=16)
    parser.add_argument('--num_workers',        type=int,   default=1)
    parser.add_argument('--pin_mem',            type=bool,  default=True)
    """

    keys = ["slide_ids", "survival_days", "survival_right_censor"]

    index   = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)
    bin_mods, extra_mods = [], []

    arg_dict = vars(args)
    for mod in ["clinical", "clinical_imputed"]:
        if arg_dict.get(mod, False): 
            extra_mods.append(mod)
    for mod in ["path_lang", "rad_lang", "path_img"]:
        if arg_dict.get(mod, False): 
            bin_mods.append(mod)

    dataset_args = {
        "data_dir": args.data_path,
        "return_key": True,
        "keys": keys,
        "label_column": args.label_col,
        "label_dtype": np.float32,
        "bin_modality_keys": bin_mods,
        "extra_modality_keys": extra_mods,
        "allow_sparse_samples": args.sparse,
        "label_fn": lambda dates: (dates < (365.0 * args.survival_years)).astype(np.float32) #predict if the patient will die in x years
    }
    loader_args = {
        "batch_size": args.batch_size,
        "pin_memory": args.pin_mem,
        "num_workers": args.num_workers,
        "collate_fn": default_collate if not args.path_img else lambda batch: collate_mixed(batch, ["label", *bin_mods, *extra_mods]),
        "persistent_workers": args.num_workers > 0,
        "drop_last": False,
    }

    train_set = MemmapDatasetMultimodal(indices=train_inds, **dataset_args)
    val_set = MemmapDatasetMultimodal(indices=validation_inds, **dataset_args)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    test_index   = np.load(f"{args.test_path}/index_arrays_labeled.npz", allow_pickle=True)
    test_args = {**dataset_args}
    test_args["data_dir"] = args.test_path
    test_set = MemmapDatasetMultimodal(**test_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    stats = [[split, len(dset), len(loader), under, over] 
             for (split, dset, loader, under, over) 
             in zip(["Train", "Valid", "Test"], 
                    [train_set, val_set, test_set], 
                    [train_loader, val_loader, test_loader],
                    [np.sum(index[args.label_col][train_inds] < args.survival_years * 365.0), np.sum(index[args.label_col][validation_inds] < args.survival_years * 365.0), np.sum(test_index[args.label_col] < args.survival_years * 365.0)],
                    [np.sum(index[args.label_col][train_inds] >= args.survival_years * 365.0), np.sum(index[args.label_col][validation_inds] >= args.survival_years * 365.0), np.sum(test_index[args.label_col] >= args.survival_years * 365.0)]
             )
            ]
    headers = ["Split", "# Samples", "# Batches", f"# Living < {args.survival_years}", f"# Living >= {args.survival_years}"]
    print(tabulate(stats, headers=headers, tablefmt="grid"),
          "\n")
    

    return train_loader, val_loader, test_loader