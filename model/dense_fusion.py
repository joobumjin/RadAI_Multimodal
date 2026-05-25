from typing import List, Dict, Optional, Union, Literal

import torch
import torch.nn as nn
from torch.nn import functional as F
from model.linear_predictor import *

class DenseFusion(nn.Module):
    """
    Logit-level fusion using an nn.Module 
    """
    def __init__(self, encoders: Dict[str, nn.Module], emb_dim: int, hidden_dims: List[int], autocast: Dict[str, bool], loss_fn: Optional[nn.Module], device: str):
        super().__init__()

        self.autocast = autocast
        self.device = device
        self.emb_dim = emb_dim
        self.loss_fn = loss_fn
        self.modality_order = list(encoders.keys())

        self.encoders = {mod: encoders[mod].to(device) for mod in encoders}

        self.pred = LinearModel(emb_dim * len(encoders), hidden_dims=hidden_dims, layer_norm=True, loss_fn=None)
        self.pred = self.pred.to(device)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:            
        logits = {}
        for modality, enc in self.encoders.items():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[modality]):
                logits[modality] = enc(x[modality]).float()
            if f"{modality}_mask" in x: 
                logits[modality] *= x[f"{modality}_mask"].view((-1, 1))
        
        catted = torch.cat([logits[mod] for mod in self.modality_order], dim=1)
        return self.pred.predict(catted)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        preds = self.predict(x)
        loss = [self.loss_fn(preds, x["label"])] if self.loss_fn is not None else []
        return preds, *loss


class DenseFusionMulti(nn.Module):
    """
    Logit-level fusion using an nn.Module 
    """
    def __init__(self, 
                 encoders: Dict[str, nn.Module], 
                 emb_dim: int, 
                 hidden_dims: List[int],
                 decoders: Dict[str, nn.Module], 
                 autocast: Dict[str, bool], 
                 loss_fn: Optional[nn.Module], 
                 device: str,
                 targets: Optional[List[str]]):
        super().__init__()

        self.autocast = autocast
        self.device = device
        self.emb_dim = emb_dim
        self.loss_fn = loss_fn
        self.modality_order = list(encoders.keys())
        self.targets = targets if targets is not None else list(self.pred.keys())

        self.encoders = {mod: encoders[mod].to(device) for mod in encoders}
        self.merge = LinearModel(emb_dim * len(encoders), hidden_dims=hidden_dims, out_dim=emb_dim, act=nn.GELU(), layer_norm=True, loss_fn=None)
        self.merge = self.merge.to(device)

        self.pred = {target: decoders[target].to(device) for target in decoders}
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:            
        logits = {}
        for modality, enc in self.encoders.items():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[modality]):
                logits[modality] = enc(x[modality]).float()
            if f"{modality}_mask" in x: 
                logits[modality] *= x[f"{modality}_mask"].view((-1, 1))
        
        merged = torch.cat([logits[mod] for mod in self.modality_order], dim=1)
        merged = self.merge.predict(merged)

        preds = {}
        for target in self.targets:
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[target]):
                preds[target] = self.pred[target].predict(merged)
            if f"{target}_mask" in x: 
                preds[target] *= x[f"{target}_mask"].view((-1, 1))
            
        return preds
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        preds = self.predict(x)

        loss = [self.loss_fn(preds, x)] if self.loss_fn is not None else []
        return preds, *loss


