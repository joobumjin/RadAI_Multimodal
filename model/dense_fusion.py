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

    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:            
        batch_size = None
        for modality in self.encoders:
            if x[modality] is not None: 
                batch_size = len(x[modality])
                break
            print(f"Dense Fusion: all None samples in batch")

        logits = {}
        for modality, enc in self.encoders.items():
            if x[modality] is not None:
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[modality]):
                    logits[modality] = enc(x[modality]).float()
            else:
                logits[modality] = torch.zeros((batch_size, self.emb_dim))
        
        catted = torch.cat([logits[mod] for mod in self.modality_order], dim=1)
        return self.pred.predict(catted)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        preds = self.predict(x)
        loss = [self.loss_fn(preds, x["label"])] if self.loss_fn is not None else []
        return preds, *loss


