from typing import List, Dict, Optional, Callable
from itertools import combinations

import torch
from torch import Tensor
from torch import nn
from model.linear_predictor import *

class AttentionFusion(nn.Module):
    """
    Keyless Attention based fusion 
    Given encoders, fuses the encoded outputs as a weighted sum
    Then predicts on weighted -> hidden_dims -> 1

    The weights for the weighted sum are learned as a shared linear emb_dim->1

    See https://ojs.aaai.org/index.php/AAAI/article/view/12319
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
        self.attn = nn.Linear(emb_dim, 1, bias=False)

        self.pred = LinearModel(emb_dim, hidden_dims=hidden_dims, batch_norm = batch_norm, layer_norm = layer_norm, rms_norm = rms_norm, loss_fn=None)
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
        for mod, enc in self.encoders.items():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[mod]):
                logits[mod] = enc(x[mod]).float()
            if f"{mod}_mask" in x: 
                logits[mod] *= x[f"{mod}_mask"].view((-1, 1))

        attn_val = {}
        for mod in self.modality_order:
            attn_val[mod] = self.attn(logits[mod]).exp()

        attn_sum = torch.stack(list(attn_val.values())).sum(0)
        attn_weights = {mod: val / attn_sum for mod, val in attn_val.items()}

        weighted = torch.stack(
            [logits[mod] * attn_weights[mod] * x[f"{mod}_mask"].view((-1, 1)) for mod in self.modality_order]
        ).sum(0)
        return self.pred.predict(weighted)
    
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        preds = self.predict(x)

        loss = [self.loss_fn(preds, x)] if self.loss_fn is not None else []
        return preds, *loss

class HierarchicalAttentionFusion(nn.Module):
    """
    Hierarchical Keyless Attention based fusion 
    Given encoders, fuses every pair of unimodal encodings using keyless attn (see above)

    Then, fuses all paired representations into a single final representation through keyless
    The fused representation is then passed to a prediction head for a final pred

    See https://www.sciencedirect.com/science/article/pii/S1746809423011023#b0275
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
        self.pairs = list(combinations(self.modality_order, 2))

        self.encoders = {mod: encoders[mod].to(device) for mod in encoders}
        self.pair_attn = {pair: nn.Linear(emb_dim, 1, bias=False, device=device) for pair in self.pairs}
        self.attn = nn.Linear(emb_dim, 1, bias=False)

        self.pred = LinearModel(emb_dim, hidden_dims=hidden_dims, batch_norm = batch_norm, layer_norm = layer_norm, rms_norm = rms_norm, loss_fn=None)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def predict(self, x: Dict[str, Tensor], get_attn: bool = False, return_mods: bool = False) -> Dict[str, Tensor]:
        logits = {}
        for modality, enc in self.encoders.items():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[modality]):
                logits[modality] = enc(x[modality]).float()
            if f"{modality}_mask" in x: 
                logits[modality] *= x[f"{modality}_mask"].view((-1, 1))

        pair_embs = {} #tuple: tensor (modality pair: combined embedding)
        inp_weights = {}
        for pair in self.pairs:
            a = self.pair_attn[pair](logits[pair[0]]).exp() 
            b = self.pair_attn[pair](logits[pair[1]]).exp()
            s = a + b
            a, b = a/s, b/s

            # if f"{pair[0]}_mask" in x: a *= x[f"{pair[0]}_mask"].view((-1, 1))
            # if f"{pair[1]}_mask" in x: b *= x[f"{pair[1]}_mask"].view((-1, 1))
            pair_embs[pair] = logits[pair[0]] * a + logits[pair[1]] * b
            inp_weights[pair] = (a,b)

        pair_vals = {pair: self.attn(emb).exp() for pair, emb in pair_embs.items()}
        val_sum = torch.stack(list(pair_vals.values())).sum(0)
        pair_vals = {pair: val / val_sum for pair, val in pair_vals.items()}

        weighted = torch.stack([pair_embs[pair] * pair_vals[pair] / val_sum for pair in self.pairs]).sum(0)
        preds = self.pred.predict(weighted)

        extras = []
        if get_attn: extras += [(inp_weights, pair_vals)]
        if return_mods: extras += [logits]

        if len(extras): return preds, *extras
        return preds
    
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        preds = self.predict(x)

        loss = [self.loss_fn(preds, x)] if self.loss_fn is not None else []
        return preds, *loss, 
