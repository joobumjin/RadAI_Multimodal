from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, default_collate

class MemmapDataset(Dataset):
    """
    Slide Bag Dataset using memory-mapped binary file.
    
    OPTIMIZED:
    - Subsample BEFORE copying from memmap (critical for large slides)
    - No unnecessary .clone() calls
    - Keeps original dtype through pipeline
    """
    
    def __init__(
        self,
        data_dir: str,
        indices: Optional[List[int]] = None,
        max_instances: Optional[int] = None,
        keys: Optional[List[str]] = None,
        return_key: bool = False,
        label_column: Optional[str] = None,
        label_dtype: Optional[np.dtype] = np.float32,
        index_filename: str = "index_arrays_labeled.npz",
        bin_path: str = "path_rad_embs.dat",
        num_modalities: int = 2,
        modality_inds: Optional[List[int]] = None
    ):
        """
        Args:
            data_dir: Directory containing features.dat and index_arrays.npz
            indices: Subset of slide indices to use (for train/val split)
            augmentation: Augmentation for view 1 (and view 2 if augmentation_view2 is None)
            augmentation_view2: Optional separate augmentation for view 2 (asymmetric)
            max_instances: Max patches to sample per slide (CRITICAL: now samples BEFORE loading)
            return_key: Whether to return slide ID with each sample
        """

        self.data_dir       = data_dir
        self.max_instances  = max_instances
        self.return_key     = return_key
        self.modality_inds  = modality_inds if modality_inds is not None else [0,1]

        # Load index
        index_path = os.path.join(data_dir, index_filename)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        index = np.load(index_path, allow_pickle=True)
        
        self._offsets   = index['offsets']
        self._slide_ids = index['slide_ids']
        self._lengths   = index['combined_lengths'].astype(int)
        if modality_inds is not None: self._lengths = self._lengths[..., self.modality_inds]
        self._labels    = index[label_column].astype(label_dtype) if label_column is not None else None

        #handle keys
        keys = [index[col].tolist() for col in keys] if self.return_key and keys is not None else None
        self._keys = None
        if self.return_key:
            self._keys = [key_tuple for key_tuple in zip(*keys)] if keys is not None else self._slide_ids

        
        # Handle different ways feat_dim might be stored
        feat_dim        = index['feat_dim']
        self._feat_dim  = int(feat_dim.item() if feat_dim.ndim == 0 else feat_dim)
        
        # Handle dtype
        dtype_arr = index['dtype']
        if dtype_arr.ndim == 0:
            self._dtype_str = str(dtype_arr.item())
        else:
            self._dtype_str = str(dtype_arr[0])
        
        # Total patches
        total_patches = index['total_patches']
        self._total_patches = int(total_patches.item() if total_patches.ndim == 0 else total_patches)
        

        # Filter valid indices (offset >= 0) and apply subset
        all_valid = np.prod(self._lengths, axis=1)
        # print(f"Valid Lengths: {len(np.flatnonzero(all_valid))}")
        if self._labels is not None: all_valid *= (~np.isnan(self._labels))
        all_valid = np.flatnonzero(all_valid)
        
        if indices is not None:
            self.indices = [i for i in indices if i in all_valid]
        else:
            self.indices = list(all_valid)
        
        # Memory-map the binary file (read-only, OS handles caching)
        bin_path = os.path.join(data_dir, bin_path)
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Features file not found: {bin_path}")
        
        bin_shape = [self._total_patches, self._feat_dim, 2]
        
        self.data = np.memmap(
            bin_path,
            dtype=self._dtype_str,
            mode='r',
            shape=bin_shape
        )
    
    def __len__(self) -> int:
        return len(self.indices)
    
    @property
    def feature_dim(self) -> int:
        return self._feat_dim
    
    def __getitem__(self, idx: int) -> Union[
        Tuple[list[torch.Tensor], np.float64],
        Tuple[list[torch.Tensor], np.float64, int],
    ]:
        """
        Returns:
            view1:      bag of features [N1, D]
            slide_id:   (if return_key=True) Slide identifier
        """
        # Map to real index
        real_idx    = self.indices[idx]

        offset      = int(self._offsets[real_idx])
        lengths     = self._lengths[real_idx]
        # slide_id    = int(self._slide_ids[real_idx])
        keys        = self._keys[real_idx] if self.return_key else None

        label = self._labels[real_idx]
        
        # Handle empty/failed slides
        # if offset < 0 or np.prod(lengths) == 0 or label == np.nan:
        #     dummy = torch.zeros((1, self._feat_dim), dtype=torch.float32)
        #     if self.return_key:
        #         return dummy, None, keys
        #     return dummy, None
        
        # =================================================================
        # CRITICAL OPTIMIZATION: Subsample BEFORE copying from memmap
        # This is the biggest win - don't load 50k rows to keep 5k
        # =================================================================
        ks = lengths
        if self.max_instances:
            ks = np.min(np.vstack((ks, [self.max_instances] * len(ks))), axis=0)
        
        feats = []
        for i, k, length in zip(self.modality_inds, list(ks), list(lengths)):
            start = np.random.randint(0, length - k + 1) if k < length else 0
            features_np = np.array(self.data[offset + start : offset + start + k, :, i], copy=True)

            # Convert to torch - keep original dtype (float16 if stored as float16)
            feats.append(torch.from_numpy(features_np))

        if self.return_key:
            return feats, label, keys
        return feats, label


# =============================================================================
# Collate Functions (unchanged)
# =============================================================================
def collate_bags(
    batch: List[Union[
        Tuple[torch.Tensor, np.float64, int],
        Tuple[torch.Tensor, np.float64],
    ]]
) -> Union[
    Tuple[List[torch.Tensor], List[np.float64], List[int]],
    Tuple[List[torch.Tensor], List[np.float64]],
]:
    """
    Collate variable-size bags into lists.
    Returns 2d list [batch, modalities] of tensors
    """
    if len(batch[0]) == 3:
        feats, labels, keys = zip(*batch)
        if isinstance(keys[0], tuple):
            keys = [np.array(key) for key in zip(*keys)]
        return list(feats), list(labels), list(keys)
    else:
        feats, labels = zip(*batch)
        return list(feats), list(labels)
    
def collate_tensors(
    batch: List[Union[
        Tuple[torch.Tensor, np.float64, int],
        Tuple[torch.Tensor, np.float64],
    ]]
):
    batch = default_collate(batch)
    if len(batch) == 3: batch[2] = [key.numpy() for key in batch[2]]
    return batch


def collate_bags_padded(
    batch: List[Union[
       Tuple[torch.Tensor, np.float64],
        Tuple[torch.Tensor, np.float64, int],
    ]],
    pad_value: float = 0.0,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
]:
    """Collate with padding for batched processing."""
    has_keys = len(batch[0]) == 3
    
    if has_keys:
        views1, labels, keys = zip(*batch)
        keys = list(keys)
        if isinstance(keys[0], tuple):
            keys = [np.array(key) for key in zip(*keys)]
    else:
        views1, labels = zip(*batch)
        keys = None
    labels = torch.from_numpy(np.array(labels)).float()

    views1 = [list(feat) for feat in list(zip(*views1))] #modalities x batch size
    
    def pad_bags(bags: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_n = max(b.shape[0] for b in bags)
        D = bags[0].shape[1]
        B = len(bags)
        
        padded = torch.full((B, max_n, D), pad_value, dtype=bags[0].dtype)
        mask = torch.zeros(B, max_n, dtype=torch.bool)
        
        for i, bag in enumerate(bags):
            n = bag.shape[0]
            padded[i, :n] = bag
            mask[i, :n] = True
        
        return [padded, mask]
    
    #pass in a batch list of tensor per modality
    views1 = [pad_bags(view1) for view1 in views1] #modalities, 2 tensors
    views1 = [list(view) for view in list(zip(*views1))] #2, modality
    # v1_padded, mask1 = views1
    #returning v1padded and mask as list [modality] of tensors [batch, bag, emb_dim]
        
    if has_keys:
        return *views1, labels, keys
    return *views1, labels
