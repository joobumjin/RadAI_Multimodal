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

class MemmapDatasetMergedMultimodal(Dataset):
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
        return_key: bool = False,
        keys: Optional[List[str]] = None,
        label_column: Optional[str] = None,
        label_dtype: Optional[np.dtype] = np.float32,
        index_filename: str = "index_arrays_labeled.npz",
        bin_path: str = "path_rad_embs.dat",
        bin_modality_keys: Optional[List[str]] = ["path_lang", "rad_lang"],
        extra_modality_keys: Optional[List[str]] = [], #eg ['clinical']
        allow_sparse_samples: bool = False
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

        bin_inds = {"path_lang": 0, "rad_lang": 1}

        self.data_dir       = data_dir
        self.max_instances  = max_instances
        self.return_key     = return_key
        self.modality_inds  = {key: bin_inds[key] for key in bin_modality_keys}
        self.sparse         = allow_sparse_samples

        # Load index
        index_path = os.path.join(data_dir, index_filename)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        index = np.load(index_path, allow_pickle=True)
        
        self._offsets   = index['offsets']
        self._slide_ids = index['slide_ids']
        self._lengths   = index['combined_lengths'].astype(int)[..., [bin_inds[key] for key in bin_modality_keys]]
        self._labels    = index[label_column].astype(label_dtype) if label_column is not None else None

        #handle keys
        self._keys = None
        if self.return_key:
            self._keys = {col: index[col] for col in keys} if keys is not None else {'slide_ids': self._slide_ids}

        self.extras = {key: index[key] for key in extra_modality_keys} if extra_modality_keys is not None else {}
        
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
        
        #Filtering
        all_valid = np.zeros((len(self._slide_ids), )) if self.sparse else np.ones((len(self._slide_ids), ))
        
        #filter labels
        if self._labels is not None: all_valid *= (~np.isnan(self._labels))

        #filter bin modalities
        if self._lengths is not None:
            if self.sparse:
                all_valid += np.any(self._lengths > 0, axis=1)  
            else:
                all_valid *= np.all(self._lengths > 0, axis=1)
                
        #filter extras
        if len(self.extras):
            for feats in self.extras.values():
                if self.sparse:
                    all_valid += ~np.isnan(feats).any(axis=1)
                else:
                    all_valid *= ~np.isnan(feats).any(axis=1)
        
        #filter keys
        if self.return_key:
            for feats in self._keys.values():
                all_valid *= ~np.isnan(feats)

        all_valid = np.flatnonzero(all_valid)
        
        if indices is not None:
            self.indices = [i for i in indices if i in all_valid]
        else:
            self.indices = list(all_valid)
        
        # Memory-map the binary file (read-only, OS handles caching)
        bin_path = os.path.join(data_dir, bin_path)
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Features file not found: {bin_path}")
        
        bin_shape = tuple([self._total_patches, self._feat_dim, 2])
        
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
        label = self._labels[real_idx:real_idx+1]
        
        sample = {"label": label}

        if self._lengths is not None:
            offset      = int(self._offsets[real_idx])
            lengths     = self._lengths[real_idx] 

            ks = lengths
            if self.max_instances:
                ks = np.min(np.vstack((ks, [self.max_instances] * len(ks))), axis=0)

            for (bin_name, bin_ind), k, length in zip(self.modality_inds.items(), list(ks), list(lengths)):
                if self.sparse and length == 0:
                    sample[f"{bin_name} mask"] = 0
                    sample[bin_name] = np.zeros((self._feat_dim), dtype=self.data.dtype)
                else:
                    sample[f"{bin_name} mask"] = 1
                    start = np.random.randint(0, length - k + 1) if k < length else 0
                    sample[bin_name] = np.array(self.data[offset + start : offset + start + k, :, bin_ind], copy=True).squeeze(0)

        for mod_name, mod_data in self.extras.items():
            data = mod_data[real_idx]
            if self.sparse and np.isnan(data).any():
                sample[f"{mod_name} mask"] = 0
                if len(mod_data.shape) == 1:
                    sample[mod_name] = 0
                else:
                    sample[mod_name] = np.zeros((mod_data.shape[1:]), dtype=mod_data.dtype)
            else:
                sample[f"{mod_name} mask"] = 1
                sample[mod_name] = data
        
        if self.return_key: 
            keys    = {key: self._keys[key][real_idx] for key in self._keys}
            sample  = {**sample, **keys}

        return sample
    
