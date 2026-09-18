from typing import List, Dict, Optional, Callable
from itertools import combinations

import torch
from torch import Tensor
from torch import nn
from model.linear_predictor import *

class AttentionFusion(nn.Module):
    """
    Logit-level fusion using an nn.Module 
    """
    def __init__(self, 
                 encoders: Dict[str, nn.Module], 
                 device: torch.device,
                 emb_dim: int, hidden_dims: List[int], 
                 autocast: Dict[str, bool], 
                 loss_fn: Optional[Callable], 
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 rms_norm: bool = False,):
        super().__init__()

        self.autocast = autocast
        self.device = device
        self.emb_dim = emb_dim
        self.loss_fn = loss_fn
        self.modality_order = list(encoders.keys())

        self.encoders = {mod: encoders[mod].to(device) for mod in encoders}
        self.attn = nn.Linear(emb_dim, 1, bias=False, device=device)

        self.pred = LinearModel(emb_dim, hidden_dims=hidden_dims, batch_norm = batch_norm, layer_norm = layer_norm, rms_norm = rms_norm, loss_fn=None)
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

    def predict(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:            
        #at it's simplest, we could just learn a scalar attention weight per modality
        #and call the attended output embedding the scaled sums of the input embeddings

        #i think hierarchal keyless attn does lambda: emb_dim -> 1 
        # lambda_i = exp(w_i@emb_i) / sum_j(exp(w @ emb_j)) with shared weight w_i
        # and then attended = sum(lambda_i * emb_i)
        #but then why not just do standard cross attention?
        # attended_1,2 = dot(proj(emb_1), proj(emb_2)) (emb_1 + emb_2) -> this sounds really weird because we dont have a sequence of tokens
        # actually, this is p similar to "keyless" https://ojs.aaai.org/index.php/AAAI/article/view/12319

        logits = {}
        attn_val = {}
        for modality, enc in self.encoders.items():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[modality]):
                logits[modality] = enc(x[modality]).float()
                attn_val[modality] = self.attn(logits[modality]).exp()
            if f"{modality}_mask" in x: 
                logits[modality] *= x[f"{modality}_mask"].view((-1, 1))
                attn_val[modality] *= x[f"{modality}_mask"].view((-1, 1))

        attn_sum = torch.stack(list(attn_val.values())).sum(0)
        for mod in attn_val: attn_val[mod] /= attn_sum

        weighted = torch.stack([logits[mod] * attn_val[mod] for mod in self.modality_order]).sum(0)
        return self.pred.predict(weighted)
    
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        preds = self.predict(x)

        loss = [self.loss_fn(preds, x)] if self.loss_fn is not None else []
        return preds, *loss

class HierarchalAttentionFusion(nn.Module):
    """
    Logit-level fusion using an nn.Module 
    """
    def __init__(self, 
                 encoders: Dict[str, nn.Module], 
                 device: torch.device,
                 emb_dim: int, hidden_dims: List[int], 
                 autocast: Dict[str, bool], 
                 loss_fn: Optional[Callable], 
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 rms_norm: bool = False,):
        super().__init__()

        self.autocast = autocast
        self.device = device
        self.emb_dim = emb_dim
        self.loss_fn = loss_fn
        self.modality_order = list(encoders.keys())
        self.pairs = combinations(self.modality_order, 2)

        self.encoders = {mod: encoders[mod].to(device) for mod in encoders}
        self.pair_attn = {pair: nn.Linear(emb_dim, 1, bias=False, device=device) for pair in self.pairs}
        self.attn = nn.Linear(emb_dim, 1, bias=False, device=device)

        self.pred = LinearModel(emb_dim, hidden_dims=hidden_dims, batch_norm = batch_norm, layer_norm = layer_norm, rms_norm = rms_norm, loss_fn=None)
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

    def predict(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        logits = {}
        for modality, enc in self.encoders.items():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[modality]):
                logits[modality] = enc(x[modality]).float()
            if f"{modality}_mask" in x: 
                logits[modality] *= x[f"{modality}_mask"].view((-1, 1))

        pair_embs = {} #tuple: tensor (modality pair: combined embedding)
        for pair in self.pairs:
            a = self.pair_attn[pair](logits[pair[0]]).exp() 
            b = self.pair_attn[pair](logits[pair[1]]).exp()
            pair_embs[pair] = (logits[pair[0]] * a + logits[pair[1]] * b) / (a+b)

        pair_vals = {pair: self.attn(pair).exp() for pair in pair_embs}
        val_sum = torch.stack(list(pair_vals.values())).sum(0)

        weighted = torch.stack([pair_embs[pair] * pair_vals[pair] / val_sum for pair in self.pairs]).sum(0)
        return self.pred.predict(weighted)
    
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        preds = self.predict(x)

        loss = [self.loss_fn(preds, x)] if self.loss_fn is not None else []
        return preds, *loss
