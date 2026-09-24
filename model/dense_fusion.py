from typing import List, Dict, Optional, Callable

import torch
from torch import Tensor
from torch import nn
from model.linear_predictor import *

class DenseFusion(nn.Module):
    """
    Intermediate fusion on unimodal encoders
    Given encoders, concatenates the encoded outputs into catted
    Then predicts on catted -> hidden_dims -> 1
    """
    def __init__(self, 
                 encoders: Dict[str, nn.Module], 
                 device: torch.device,
                 emb_dim: int, 
                 hidden_dims: List[int], 
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

        self.encoders = nn.ModuleDict(encoders)

        self.pred = LinearModel(emb_dim * len(encoders), hidden_dims=hidden_dims, act = nn.GELU(), 
                                batch_norm = batch_norm, layer_norm = layer_norm, rms_norm = rms_norm, loss_fn=None)
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
        catted = [logits[mod] for mod in self.modality_order]    
        catted = torch.cat(catted, dim=1)
        return self.pred.predict(catted)
    
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        preds = self.predict(x)

        loss = [self.loss_fn(preds, x)] if self.loss_fn is not None else []
        return preds, *loss


class DenseFusionMulti(nn.Module):
    """
    Intermediate fusion on unimodal encoders with multi target training regime (UNUSED)
    Intended for multiple prediction targets
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

    def predict(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:            
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
    
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        preds = self.predict(x)

        loss = [self.loss_fn(preds, x)] if self.loss_fn is not None else []
        return preds, *loss


class SingleModAE(nn.Module):
    """
    Unimodal AE (UNUSED)
    Intended to contribute reconstruction loss to a multi target training regime
    """
    def __init__(self, mod_name: str, enc: nn.Module, dec: nn.Module, device: torch.device, autocast: Dict[str, bool], loss_fn = None):
        super().__init__()
        self.mod = mod_name
        self.autocast = autocast
        self.loss_fn = loss_fn
        self.ae = nn.Sequential(enc, dec)
        self.device = device
        
    def predict(self, x):
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.autocast[self.mod]):
            preds = self.ae(x[self.mod])
        if f"{self.mod}_mask" in x: 
            preds *= x[f"{self.mod}_mask"].view((-1, 1))
            
        return preds        

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        preds = self.predict(x)

        lv = self.loss_fn(preds, x[self.mod]) if self.loss_fn is not None else None
        loss = [{"total_loss": lv, self.mod: lv.detach().cpu()}] if self.loss_fn is not None else []
        return preds, *loss
