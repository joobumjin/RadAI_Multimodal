from argparse import Namespace
from functools import reduce
from typing import Optional

import numpy as np
from tabulate import tabulate

from data import *

def get_loaders(args: Namespace, 
                train_inds, 
                validation_inds, 
                keys: list[str] = ["slide_ids", "survival_days", "survival_right_censor"],
                label_key = "label"):
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
        "label_fn": lambda dates: (dates < (365.0 * args.survival_years)).astype(np.float32), #predict if the patient will die in x years,
        "label_key": label_key
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
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    if len(validation_inds):
        val_set = MemmapDatasetMultimodal(indices=validation_inds, **dataset_args)
        val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    else:
        val_set, val_loader = None, None

    test_index = np.load(f"{args.test_path}/index_arrays_labeled.npz", allow_pickle=True)
    test_args = {**dataset_args}
    test_args["data_dir"] = args.test_path
    test_set = MemmapDatasetMultimodal(**test_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    combine = (lambda x, y: x & y) if not args.sparse else (lambda x, y: x | y)
    test_mask = ~test_index["excluded"] & reduce(combine, [test_index[f"{mod}_mask"] for mod in bin_mods + extra_mods])

    stats = [[split, len(dset), len(loader), under, over] 
             for (split, dset, loader, under, over) 
             in zip(["Train", "Test"], 
                    [train_set, test_set], 
                    [train_loader, test_loader],
                    [np.sum(index[args.label_col][train_inds] < (args.survival_years * 365.0)), np.sum(test_index[args.label_col][test_mask] < (args.survival_years * 365.0))],
                    [np.sum(index[args.label_col][train_inds] >= (args.survival_years * 365.0)), np.sum(test_index[args.label_col][test_mask] >= (args.survival_years * 365.0))])
            ]
    if len(validation_inds):
        valid_stats = ["Valid", len(val_set), len(val_loader), np.sum(index[args.label_col][validation_inds] < (args.survival_years * 365.0)), np.sum(index[args.label_col][validation_inds] >= (args.survival_years * 365.0))] 
        stats = [stats[0], valid_stats, stats[1]]

    headers = ["Split", "# Samples", "# Batches", f"# Living < {args.survival_years}yr", f"# Living >= {args.survival_years} yr"]
    print(tabulate(stats, headers=headers, tablefmt="grid"), "\n")

    return train_loader, val_loader, test_loader


def get_key_loaders(args: Namespace, 
                    train_inds, validation_inds, 
                    keys: list[str] = ["slide_ids", "survival_days",],
                    sparse = True):
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
        "bin_modality_keys": bin_mods,
        "extra_modality_keys": extra_mods,
        "allow_sparse_samples": sparse,
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
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    if len(validation_inds):
        val_set = MemmapDatasetMultimodal(indices=validation_inds, **dataset_args)
        val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    else:
        val_set, val_loader = None, None

    test_args = {**dataset_args}
    test_args["data_dir"] = args.test_path
    test_set = MemmapDatasetMultimodal(**test_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    stats = [[split, len(dset), len(loader)] 
             for (split, dset, loader) 
             in zip(["Train", "Test"], 
                    [train_set, test_set], 
                    [train_loader, test_loader])
            ]
    if len(validation_inds):
        valid_stats = ["Valid", len(val_set), len(val_loader), np.sum(index[args.label_col][validation_inds] < (args.survival_years * 365.0)), np.sum(index[args.label_col][validation_inds] >= (args.survival_years * 365.0))] 
        stats = [stats[0], valid_stats, stats[1]]

    headers = ["Split", "# Samples", "# Batches"]
    print(tabulate(stats, headers=headers, tablefmt="grid"), "\n")

    return train_loader, val_loader, test_loader



def get_input_loader(args: Namespace, bin_mods, extra_mods, train_inds, validation_inds):
    """
    Required Args: 
    parser.add_argument('--data_path',          type=str,   default="../updated_multimodal_bins")

    parser.add_argument('--label_col',          type=str,   default="survival_days")

    parser.add_argument('--sparse',             action="store_true")

    parser.add_argument('--batch_size',         type=int,   default=16)
    parser.add_argument('--num_workers',        type=int,   default=1)
    parser.add_argument('--pin_mem',            type=bool,  default=True)
    """

    index   = np.load(f"{args.data_path}/index_arrays_labeled.npz", allow_pickle=True)

    dataset_args = {
        "data_dir": args.data_path,
        "return_key": True,
        "keys": [],
        "label_column": None,
        "label_dtype": np.float32,
        "bin_modality_keys": bin_mods,
        "extra_modality_keys": extra_mods,
        "allow_sparse_samples": args.sparse,
    }
    loader_args = {
        "batch_size": args.batch_size,
        "pin_memory": args.pin_mem,
        "num_workers": args.num_workers,
        "collate_fn": default_collate,
        "persistent_workers": args.num_workers > 0,
        "drop_last": False,
    }

    train_set = MemmapDatasetMultimodal(indices=train_inds, **dataset_args)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    if len(validation_inds):
        val_set = MemmapDatasetMultimodal(indices=validation_inds, **dataset_args)
        val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    else:
        val_set, val_loader = None, None

    test_index = np.load(f"{args.test_path}/index_arrays_labeled.npz", allow_pickle=True)
    test_args = {**dataset_args}
    test_args["data_dir"] = args.test_path
    test_set = MemmapDatasetMultimodal(**test_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    combine = (lambda x, y: x & y) if not args.sparse else (lambda x, y: x | y)
    test_mask = ~test_index["excluded"] & reduce(combine, [test_index[f"{mod}_mask"] for mod in bin_mods + extra_mods])

    stats = [[split, len(dset), len(loader)] 
             for (split, dset, loader) 
             in zip(["Train", "Test"], 
                    [train_set, test_set], 
                    [train_loader, test_loader])
            ]
    if len(validation_inds):
        valid_stats = ["Valid", len(val_set), len(val_loader), np.sum(index[args.label_col][validation_inds] < (args.survival_years * 365.0)), np.sum(index[args.label_col][validation_inds] >= (args.survival_years * 365.0))] 
        stats = [stats[0], valid_stats, stats[1]]

    headers = ["Split", "# Samples", "# Batches"]
    print(tabulate(stats, headers=headers, tablefmt="grid"), "\n")

    return train_loader, val_loader, test_loader