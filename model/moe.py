from typing import List, Dict, Union, Literal, Iterable
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn import functional as F

class MoE(nn.Module):
    """
    Usually, a Mixture of Experts maps a single input to k experts, using a weighted sum to combine the expert predictions

    """
    def __init__(self, experts: List[nn.Module], pred_dim: int, modality_order: List[str], autocast: Dict[str, bool], fused_input_dim: int, topk: int, loss_fn: nn.Module, device: str):
        super().__init__()

        self.autocast = autocast
        self.device = device
        self.loss_fn = loss_fn
        self.modality_order = modality_order

        self.experts = {mod: experts[mod].to(device) for mod in experts} #each 
        self.pred_dim = pred_dim
        self.topk = topk

        self.gating = nn.Linear(fused_input_dim, len(self.modality_order), bias=False)
        self.gating = self.gating.to(device)

    def predict(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:            
        #fuse input modalities
        fused = torch.cat([x[modality].to(torch.float16()) for modality in self.modality_order], dim = 1)
        
        gate_scores = F.softmax(self.gating(fused), dim=-1)
        # Select top-k experts
        # batch x topk
        top_k_values, top_k_indices = torch.topk(gate_scores, self.topk, dim=-1)

        # Combine outputs from selected experts
        output = torch.zeros(fused.shape[0], self.pred_dim)
        for i in range(self.top_k): #for each topk
            #get the relevant experts for ith topk
            expert_inds = top_k_indices[:, i]
            # there's probably some way to parallelize this by re-batching based on chosen experts
            expert_out = torch.stack([self.experts[expert_ind](fused[sample_ind]) for sample_ind, expert_ind in enumerate(expert_inds)])
            #scale by weights
            output += expert_out * top_k_values[:, i].unsqueeze(-1)
        
        return output
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        preds = self.predict(x)
        loss = self.loss_fn(preds, x["label"])
        return loss, preds


