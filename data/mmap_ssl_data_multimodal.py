from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# =============================================================================
# Bag Augmentations (OPTIMIZED - lazy float32, no unnecessary copies)
# =============================================================================
@dataclass
class BagAugmentationConfig:
    """Configuration for bag-level augmentations."""
    
    subsample_ratio: Tuple[float, float] = (0.5, 0.9)
    instance_dropout_prob: float = 0.1
    noise_std: float = 0.1
    feature_dropout_prob: float = 0.05
    shuffle: bool = True
    min_instances: int = 16
    
    def __post_init__(self):
        if not (0 < self.subsample_ratio[0] <= self.subsample_ratio[1] <= 1.0):
            raise ValueError(f"Invalid subsample_ratio: {self.subsample_ratio}")
        if not (0 <= self.instance_dropout_prob < 1.0):
            raise ValueError(f"Invalid instance_dropout_prob: {self.instance_dropout_prob}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be non-negative: {self.noise_std}")
        if not (0 <= self.feature_dropout_prob < 1.0):
            raise ValueError(f"Invalid feature_dropout_prob: {self.feature_dropout_prob}")
        if self.min_instances < 1:
            raise ValueError(f"min_instances must be >= 1: {self.min_instances}")


class BagAugmentation:
    """
    Augmentations for bags of patch features.
    
    OPTIMIZED: 
    - Only upcast to float32 when doing noise/dropout operations
    - Avoids unnecessary copies
    - Returns original dtype
    """
    
    def __init__(self, config: Optional[BagAugmentationConfig] = None):
        self.config = config or BagAugmentationConfig()
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a bag of features.
        
        Args:
            features: [N, D] patch features (will be modified in-place for index ops)
            
        Returns:
            Augmented features [N', D] where N' <= N
        """
        cfg = self.config
        
        if features.ndim != 2:
            raise ValueError(f"Expected 2D tensor [N, D], got shape {features.shape}")
        
        N, D = features.shape
        device = features.device
        original_dtype = features.dtype
        
        # Check if we need float32 for numerical operations
        needs_float32 = (cfg.noise_std > 0) or (cfg.feature_dropout_prob > 0)
        
        # Handle edge case: very small bags
        if N <= cfg.min_instances:
            if needs_float32:
                features = features.float()
                if cfg.noise_std > 0:
                    features = features + torch.randn_like(features) * cfg.noise_std
                if cfg.feature_dropout_prob > 0:
                    feat_mask = torch.rand(D, device=device) > cfg.feature_dropout_prob
                    features = features * feat_mask.float().unsqueeze(0)
                return features.to(original_dtype)
            return features
        
        # 1. Subsample: keep random fraction of instances (index operation, no cast needed)
        if cfg.subsample_ratio[1] < 1.0 or cfg.subsample_ratio[0] < 1.0:
            ratio = random.uniform(cfg.subsample_ratio[0], cfg.subsample_ratio[1])
            n_keep = max(cfg.min_instances, int(N * ratio))
            n_keep = min(n_keep, N)
            
            if n_keep < N:
                indices = torch.randperm(N, device=device)[:n_keep]
                features = features[indices]
                N = features.shape[0]
        
        # 2. Instance dropout: randomly drop instances (index operation, no cast needed)
        if cfg.instance_dropout_prob > 0 and N > cfg.min_instances:
            keep_mask = torch.rand(N, device=device) > cfg.instance_dropout_prob
            n_kept = keep_mask.sum().item()
            
            if n_kept < cfg.min_instances:
                false_indices = (~keep_mask).nonzero(as_tuple=True)[0]
                n_needed = cfg.min_instances - int(n_kept)
                if len(false_indices) >= n_needed:
                    extra_keep = false_indices[torch.randperm(len(false_indices), device=device)[:n_needed]]
                    keep_mask[extra_keep] = True
            
            if keep_mask.sum() > 0:
                features = features[keep_mask]
                N = features.shape[0]
        
        # 3. Shuffle order (index operation, no cast needed)
        if cfg.shuffle and N > 1:
            perm = torch.randperm(N, device=device)
            features = features[perm]
        
        # 4 & 5: Only cast to float32 if we need noise or feature dropout
        if needs_float32:
            features = features.float()
            
            # Add Gaussian noise
            if cfg.noise_std > 0:
                features = features + torch.randn_like(features) * cfg.noise_std
            
            # Feature dimension dropout
            if cfg.feature_dropout_prob > 0:
                feat_mask = torch.rand(D, device=device) > cfg.feature_dropout_prob
                features = features * feat_mask.float().unsqueeze(0)
            
            return features.to(original_dtype)
        
        return features
    
    @classmethod
    def weak(cls) -> "BagAugmentation":
        """Weak augmentation (for BYOL target or evaluation)."""
        return cls(BagAugmentationConfig(
            subsample_ratio=(0.8, 0.95),
            instance_dropout_prob=0.02,
            noise_std=0.02,
            feature_dropout_prob=0.01,
            min_instances=32,
        ))
    
    @classmethod
    def strong(cls) -> "BagAugmentation":
        """Strong augmentation (for contrastive learning)."""
        return cls(BagAugmentationConfig(
            subsample_ratio=(0.4, 0.8),
            instance_dropout_prob=0.15,
            noise_std=0.15,
            feature_dropout_prob=0.1,
            min_instances=16,
        ))
    
    @classmethod
    def medium(cls) -> "BagAugmentation":
        """Medium augmentation (default, balanced)."""
        return cls(BagAugmentationConfig(
            subsample_ratio=(0.5, 0.9),
            instance_dropout_prob=0.1,
            noise_std=0.1,
            feature_dropout_prob=0.05,
            min_instances=16,
        ))
    
    @classmethod
    def index_only(cls) -> "BagAugmentation":
        """Index-only augmentation (no noise, no dropout - stays in original dtype)."""
        return cls(BagAugmentationConfig(
            subsample_ratio=(0.5, 0.9),
            instance_dropout_prob=0.1,
            noise_std=0.0,  # No noise = no float32 needed
            feature_dropout_prob=0.0,  # No dropout = no float32 needed
            min_instances=16,
        ))


class MemmapSSLDataset(Dataset):
    """
    SSL Dataset using memory-mapped binary file.
    
    OPTIMIZED:
    - Subsample BEFORE copying from memmap (critical for large slides)
    - No unnecessary .clone() calls
    - Keeps original dtype through pipeline
    """
    
    def __init__(
        self,
        data_dir: str,
        indices: Optional[List[int]] = None,
        augmentation: Optional[BagAugmentation] = None,
        augmentation_view2: Optional[BagAugmentation] = None,
        max_instances: Optional[int] = None,
        return_key: bool = False,
        label_column: Optional[str] = None,
        index_filename: str = "index_arrays.npz",
        bin_path: str = "path_rad_embs.dat",
        num_modalities: int = 2,
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
        
        # Load index
        index_path = os.path.join(data_dir, index_filename)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        index = np.load(index_path, allow_pickle=True)
        
        self._offsets   = index['offsets']
        self._lengths   = index['combined_lengths'].astype(int)
        self._slide_ids = index['slide_ids']
        self._labels    = index[label_column] if label_column is not None else None
        
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
        # can replace with a flatnonzero
        # all_valid = [i for i in range(len(self._offsets)) if self._offsets[i] >= 0]
        all_valid = np.flatnonzero(np.prod(self._lengths, axis=1))
        
        if indices is not None:
            self.indices = [i for i in indices if i in all_valid or np.prod(self._lengths[i]) >= 0]
        else:
            self.indices = list(all_valid)
        
        # Memory-map the binary file (read-only, OS handles caching)
        bin_path = os.path.join(data_dir, bin_path)
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Features file not found: {bin_path}")
        
        bin_shape = [self._total_patches, self._feat_dim]
        if num_modalities > 1: bin_shape += [num_modalities]
        
        self.data = np.memmap(
            bin_path,
            dtype=self._dtype_str,
            mode='r',
            shape=bin_shape
        )
        
        # Augmentations
        self.augmentation = augmentation or BagAugmentation.medium()
        self.augmentation_view2 = augmentation_view2
    
    def __len__(self) -> int:
        return len(self.indices)
    
    @property
    def feature_dim(self) -> int:
        return self._feat_dim
    
    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, str],
    ]:
        """
        Returns:
            view1: Augmented bag [N1, D]
            view2: Differently augmented bag [N2, D]
            slide_id: (if return_key=True) Slide identifier
        """
        # Map to real index
        real_idx    = self.indices[idx]

        offset      = int(self._offsets[real_idx])
        lengths     = self._lengths[real_idx]
        slide_id    = int(self._slide_ids[real_idx])

        label = self._labels[real_idx]
        
        # Handle empty/failed slides
        if offset < 0 or np.prod(lengths) == 0 or label == np.nan:
            dummy = torch.zeros((1, self._feat_dim), dtype=torch.float32)
            if self.return_key:
                return dummy, None, slide_id
            return dummy, None
        
        # =================================================================
        # CRITICAL OPTIMIZATION: Subsample BEFORE copying from memmap
        # This is the biggest win - don't load 50k rows to keep 5k
        # =================================================================
        ks = lengths
        if self.max_instances:
            ks = np.min(np.vstack((ks, [self.max_instances] * len(ks))), axis=0)
        
        feats = []
        for i, (k, length) in enumerate(zip(list(ks), list(lengths))):
            start = np.random.randint(0, length - k + 1) if k < length else 0
            features_np = np.array(self.data[offset + start : offset + start + k, :, i], copy=True)

            # Convert to torch - keep original dtype (float16 if stored as float16)
            feats.append(torch.from_numpy(features_np))
        # =================================================================
        # CRITICAL OPTIMIZATION: No .clone() - augmentation doesn't need it
        # The augmentation only does indexing ops (which create new tensors)
        # and additive ops (which also create new tensors)
        # =================================================================
        view1 = [self.augmentation(feat) for feat in feats]
        
        aug2 = self.augmentation_view2 if self.augmentation_view2 is not None else self.augmentation
        view2 =  [aug2(feat) for feat in feats]
        
        if self.return_key:
            return view1, view2, slide_id
        return view1, view2


# =============================================================================
# Collate Functions (unchanged)
# =============================================================================
def collate_ssl_bags(
    batch: List[Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, str],
    ]]
) -> Union[
    Tuple[List[torch.Tensor], List[torch.Tensor]],
    Tuple[List[torch.Tensor], List[torch.Tensor], List[str]],
]:
    """Collate variable-size bags into lists."""
    if len(batch[0]) == 3:
        views1, views2, keys = zip(*batch)
        return [list(view1) for view1 in list(zip(*views1))], [list(view2) for view2 in list(zip(*views2))], list(keys)
    else:
        views1, views2 = zip(*batch)
        return [list(view1) for view1 in list(zip(*views1))], [list(view2) for view2 in list(zip(*views2))]


def collate_ssl_bags_padded(
    batch: List[Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, str],
    ]],
    pad_value: float = 0.0,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
]:
    """Collate with padding for batched processing."""
    has_keys = len(batch[0]) == 3
    
    if has_keys:
        views1, views2, keys = zip(*batch)
        keys = list(keys)
    else:
        views1, views2 = zip(*batch)
        keys = None
    views1 = [list(feat) for feat in list(zip(*views1))] #modalities x batch size
    views2 = [list(feat) for feat in list(zip(*views2))] #modalities x batch size
        
    def pad_bags(bags: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_n = max(b.shape[0] for b in bags)
        D = bags[0].shape[1]
        B = len(bags)
        
        padded = torch.full((B, max_n, D), pad_value, dtype=bags[0].dtype)
        mask = torch.zeros(B, max_n, dtype=int)
        
        for i, bag in enumerate(bags):
            n = bag.shape[0]
            padded[i, :n] = bag
            mask[i, :n] = 1
        
        return [padded, mask]
    
    views1 = [pad_bags(view1) for view1 in views1] #modalities, 2 tensors
    views1 = [list(view) for view in list(zip(*views1))] #2, modality
    # v1_padded, mask1 = views1
    views2 = [pad_bags(view) for view in views2] #modalities, 2 tensors
    views2 = [list(view) for view in list(zip(*views2))] #2, modality
    # v1_padded, mask1 = views1
  
    if has_keys:
        return views1[0], views2[0], views1[1], views2[1], keys
    return views1[0], views2[0], views1[1], views2[1]