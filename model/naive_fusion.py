from typing import List, Dict, Union, Literal
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.linear_predictor import *

class Fuser(nn.Module, ABC):
    def __init__(self, modalities: List[str], **kwargs):
        super().__init__()
        self.modalities = modalities
    
    @abstractmethod
    def forward(self, x: Dict[str, torch.Tensor], masked=False) -> torch.Tensor: 
        ...

class NaiveSum(Fuser):
    def forward(self, x, masked=False):
        out =  torch.sum(torch.stack([x[mod] for mod in self.modalities]), dim=0)
        return out
    
class NaiveAvg(Fuser):
    def forward(self, x, masked=False):
        modality_count = len(self.modalities)
        if masked:
            modality_count = torch.sum(torch.stack([x[f"{mod}_mask"] for mod in self.modalities]), dim=0).view((-1, 1))
           
        return torch.sum(torch.stack([x[mod] for mod in self.modalities]), dim=0) / modality_count

class LearnedWeightSum(Fuser):
    def __init__(self, modalities: List[str], mod_dim: int = 1, out_dim: int = 1, **kwargs):
        super().__init__(modalities)
        self.weights = nn.Linear(len(modalities) * mod_dim, out_dim, bias=False)

    def forward(self, x: Dict[str, torch.Tensor], masked=False) -> torch.Tensor:
        x = torch.cat([x[mod] for mod in self.modalities], dim=1)
        return self.weights(x) #kind of a crude way to do it
    

FUSERS = {
    "naive_sum": NaiveSum,
    "naive_avg": NaiveAvg,
    "weighted_sum": LearnedWeightSum
}
    

class LogitFusion(nn.Module):
    """
    Logit-level fusion using an nn.Module 
    """
    def __init__(self, encoders: Dict[str, nn.Module], autocast: Dict[str, bool], fusion_fn: Fuser, loss_fn: nn.Module, device: str):
        super().__init__()

        self.autocast = autocast
        self.device = device
        self.loss_fn = loss_fn
        self.modality_order = list(encoders.keys())

        self.encoders = {mod: encoders[mod].to(device) for mod in encoders}

        self.fusion_fn = fusion_fn(self.modality_order)
        self.fusion_fn = self.fusion_fn.to(device)

    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:            
        logits = {}
        masked = False
        for modality, enc in self.encoders.items():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[modality]):
                logits[modality] = enc.predict(x[modality]).float()
            if f"{modality}_mask" in x: 
                masked = True
                logits[modality] *= x[f"{modality}_mask"].view((-1, 1))
                logits[f"{modality}_mask"] = x[f"{modality}_mask"]
        
        return self.fusion_fn(logits, masked=masked)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        preds = self.predict(x)
        loss = self.loss_fn(preds, x["label"])
        return loss, preds


class EmbFusion(nn.Module):
    """
    Embedding-level fusion using an nn.Module 
    """
    def __init__(self, encoders: Dict[str, nn.Module], emb_dim: int, hidden_dims: List[int], autocast: Dict[str, bool], fusion_fn: Fuser, loss_fn: Optional[nn.Module], device: str):
        super().__init__()

        self.autocast = autocast
        self.device = device
        self.loss_fn = loss_fn
        self.modality_order = list(encoders.keys())

        self.encoders = {mod: encoders[mod].to(device) for mod in encoders}

        self.fusion_fn = fusion_fn(self.modality_order, mod_dim=emb_dim, out_dim=emb_dim)
        self.fusion_fn = self.fusion_fn.to(device)

        self.pred = LinearModel(emb_dim, hidden_dims=hidden_dims, layer_norm=True, loss_fn=None)
        self.pred = self.pred.to(device)

    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:            
        logits = {}
        masked = False
        for modality, enc in self.encoders.items():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[modality]):
                logits[modality] = enc.predict(x[modality]).float()
            if f"{modality}_mask" in x: 
                masked = True
                logits[modality] *= x[f"{modality}_mask"].view((-1, 1))
                logits[f"{modality}_mask"] = x[f"{modality}_mask"]
        
        fused = self.fusion_fn(logits, masked=masked)
        return self.pred.predict(fused)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        preds = self.predict(x)
        loss = [self.loss_fn(preds, x["label"])] if self.loss_fn is not None else []
        return preds, *loss

