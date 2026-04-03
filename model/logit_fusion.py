from typing import List, Dict, Union, Literal

import torch
import torch.nn as nn
from torch.nn import functional as F

NAIVE_FUSERS = Literal["sum", "avg"]
fusion_fns = {
    "sum": lambda modality_logits: torch.sum(torch.stack(list(modality_logits.values())), axis=-1),
    "avg": lambda modality_logits: torch.sum(torch.stack(list(modality_logits.values())), axis=-1) / len(modality_logits)
}

class LearnedWeightSum(nn.Module):
    def __init__(self, modalities: List[str]):
        super().__init__()

        self.modalities = modalities        
        self.weights = nn.Linear(len(modalities), 1, bias=False)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.stack([x[mod] for mod in self.modalities]).T
        return self.weights(x) #kind of a crude way to do it
    

class LogitFusion(nn.Module):
    def __init__(self, encoders: Dict[str, nn.Module], fusion_fn: Union[NAIVE_FUSERS, nn.Module], loss_fn: nn.Module):
        super().__init__()

        self.encoders = encoders
        self.loss_fn = loss_fn
        self.modality_order = encoders.keys()

        if issubclass(fusion_fn, nn.Module): self.fusion_fn = fusion_fn(self.modality_order)
        elif isinstance(fusion_fn, NAIVE_FUSERS): self.fusion_fn = fusion_fns[fusion_fn]
        else:
            print(f"Unknown Logit Fusion Operation {fusion_fn}")

    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:            
        logits = {modality: enc.pred(x[modality]) for modality, enc in self.encoders.items()}

        return self.fusion_fn(logits)
    
    def forward(self, x: Dict[str, torch.Tensor], y: torch.Tensor) -> Dict[str, torch.Tensor]:
        preds = self.predict(x)

        loss = self.loss_fn(x, y)

        return {"preds": preds, "loss": loss}


