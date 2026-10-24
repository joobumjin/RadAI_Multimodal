from argparse import Namespace

from tabulate import tabulate
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader

from data import *
from util.consts import *

def get_datasets(
        args: Namespace,
        keys: list[str] = ["slide_ids", "survival_days", "survival_right_censor"],
    )-> tuple[MemmapDatasetMultimodal, MemmapDatasetMultimodal]:
    """
    Returns the train and test Datasets as dictated by args

    Parameters
    ----------
    args : Namespace
        parsed CLI arguments
    keys : list[str]
        other keys to get per sample 

    Returns
    ----------
    train_set : MemmapDatasetMultimodal
        Dataset at the training dataset path, as dictated by args
    test_set : MemmapDatasetMultimodal
        Dataset at the testing dataset path, as dictated by args
    """

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
        "allow_sparse_samples": args.sparse,
    }
    train_set = MemmapDatasetMultimodal(**dataset_args)

    test_args = {**dataset_args}
    test_args["data_dir"] = args.test_path
    test_set = MemmapDatasetMultimodal(**test_args)

    return train_set, test_set

# --------------------------------------------------------

def get_combined_loaders(
        args: Namespace, 
        dataset: MemmapDatasetMultimodal,
        train_inds: np.ndarray, 
        validation_inds: np.ndarray, 
        test_inds: np.ndarray,
        verbose: bool = False
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Given a combined dataset, splits and returns train, val, and test laoders

    Parameters
    ----------
    args : Namespace
        parsed CLI arguments
    dataset : MemmapDatasetMultimodal
        a combined (train, val, test) dataset. usually a ConcatDataset instance
    train_inds : np.ndarray
        indices of train samples inside dataset
    validation_inds : np.ndarray
        indices of validation samples inside dataset
    test_inds : np.ndarray
        indices of test samples inside dataset
    verbose : bool
        whether or not to print dataset details by split

    Returns
    ----------
    train_loader : DataLoader
        DataLoader for train data
    val_loader : DataLoader
        DataLoader for validation data
    test_loader : DataLoader
        DataLoader for test data
    """

    bin_mods, extra_mods = [], []

    arg_dict = vars(args)
    for mod in ["clinical", "clinical_imputed"]:
        if arg_dict.get(mod, False): 
            extra_mods.append(mod)
    for mod in ["path_lang", "rad_lang", "path_img"]:
        if arg_dict.get(mod, False): 
            bin_mods.append(mod)

    loader_args = {
        "batch_size": args.batch_size,
        "pin_memory": args.pin_mem,
        "num_workers": args.num_workers,
        "collate_fn": default_collate if not args.path_img else lambda batch: collate_mixed(batch, ["label", *bin_mods, *extra_mods]),
        "persistent_workers": args.num_workers > 0,
        "drop_last": False,
    }

    train_set = Subset(dataset, train_inds)
    train_set._dtypes = dataset._dtypes
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    if len(validation_inds): 
        val_set = Subset(dataset, validation_inds)
        val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    else: val_loader = None

    test_set = Subset(dataset, test_inds)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    stats = [[split, len(dset), len(loader)] 
             for (split, dset, loader) 
             in zip(["Train", "Test"], 
                    [train_set, test_set], 
                    [train_loader, test_loader])]
    
    if len(validation_inds):
        valid_stats = ["Valid", len(val_set), len(val_loader)] 
        stats = [stats[0], valid_stats, stats[1]]

    headers = ["Split", "# Samples", "# Batches"]
    if verbose:
        print(tabulate(stats, headers=headers, tablefmt="grid"), "\n")

    return train_loader, val_loader, test_loader

def get_subset_loaders(
        args: Namespace, 
        tv_dataset: MemmapDatasetMultimodal,
        train_inds: np.ndarray, 
        validation_inds: np.ndarray,
        test_set: MemmapDatasetMultimodal,
        verbose: bool = False
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Given separate train+val and test datasets, splits and returns train, val, and test laoders

    Parameters
    ----------
    args : Namespace
        parsed CLI arguments
    dataset : MemmapDatasetMultimodal
        a combined (train, val) dataset
    train_inds : np.ndarray
        indices of train samples inside dataset
    validation_inds : np.ndarray
        indices of validation samples inside dataset
    test_set : MemmapDatasetMultimodal
        testing dataset
    verbose : bool
        whether or not to print dataset details by split

    Returns
    ----------
    train_loader : DataLoader
        DataLoader for train data
    val_loader : DataLoader
        DataLoader for validation data
    test_loader : DataLoader
        DataLoader for test data
    """
    
    bin_mods, extra_mods = [], []

    arg_dict = vars(args)
    for mod in ["clinical", "clinical_imputed"]:
        if arg_dict.get(mod, False): 
            extra_mods.append(mod)
    for mod in ["path_lang", "rad_lang", "path_img"]:
        if arg_dict.get(mod, False): 
            bin_mods.append(mod)

    loader_args = {
        "batch_size": args.batch_size,
        "pin_memory": args.pin_mem,
        "num_workers": args.num_workers,
        "collate_fn": default_collate if not args.path_img else lambda batch: collate_mixed(batch, ["label", *bin_mods, *extra_mods]),
        "persistent_workers": args.num_workers > 0,
        "drop_last": False,
    }

    train_set = Subset(tv_dataset, train_inds)
    train_set._dtypes = tv_dataset._dtypes
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    if len(validation_inds): 
        val_set = Subset(tv_dataset, validation_inds)
        val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    else: val_loader = None

    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    stats = [[split, len(dset), len(loader)] 
             for (split, dset, loader) 
             in zip(["Train", "Test"], 
                    [train_set, test_set], 
                    [train_loader, test_loader])]
    
    if len(validation_inds):
        valid_stats = ["Valid", len(val_set), len(val_loader)] 
        stats = [stats[0], valid_stats, stats[1]]

    headers = ["Split", "# Samples", "# Batches"]
    if verbose:
        print(tabulate(stats, headers=headers, tablefmt="grid"), "\n")

    return train_loader, val_loader, test_loader