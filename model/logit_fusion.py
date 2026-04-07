from typing import List, Dict, Union, Literal
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn import functional as F

class Fuser(nn.Module, ABC):
    def __init__(self, modalities: List[str]):
        super().__init__()
        self.modalities = modalities
    
    @abstractmethod
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor: ...

class NaiveSum(Fuser):
    def forward(self, x):
        return torch.sum(torch.stack([x[mod] for mod in self.modalities]), axis=0)
    
class NaiveAvg(Fuser):
    def forward(self, x):
        return torch.sum(torch.stack([x[mod] for mod in self.modalities]), axis=0) / len(x)

class LearnedWeightSum(Fuser):
    def __init__(self, modalities: List[str]):
        super().__init__(modalities)
        self.weights = nn.Linear(len(modalities), 1, bias=False)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.stack([x[mod] for mod in self.modalities]).T
        return self.weights(x).squeeze(axis=-1) #kind of a crude way to do it
    

class LogitFusion(nn.Module):
    """
    Logit-level fusion using an nn.Module 
    """
    def __init__(self, encoders: Dict[str, nn.Module], autocast: Dict[str, bool], fusion_fn: Fuser, loss_fn: nn.Module, device: str):
        super().__init__()

        self.encoders = encoders
        self.autocast = autocast
        self.device = device
        self.loss_fn = loss_fn
        self.modality_order = list(encoders.keys())
        self.fusion_fn = fusion_fn(self.modality_order)

    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:            
        logits = {}
        for modality, enc in self.encoders.items():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[modality]):
                logits[modality] = enc.predict(x[modality])

        return self.fusion_fn(logits)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        preds = self.predict(x)
        loss = self.loss_fn(preds, x["label"])
        return loss, preds


