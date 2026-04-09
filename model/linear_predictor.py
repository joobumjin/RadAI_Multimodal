from typing import Optional, Tuple, Union
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F
from model.modules import *
from model.losses import *

class LinearModel(nn.Module):
    def __init__(self, 
                 input_dim: int = 23,
                 hidden_dims: list[int] = [128, 128, 32],
                 out_dim: int = 1,
                 act: nn.Module = nn.LeakyReLU(),
                 loss_fn: Optional[Callable] = F.mse_loss,
                 batch_norm: bool = False
    ):
        super().__init__()
        self.loss_fn = loss_fn

        self.predictor = create_mlp(
            in_dim      = input_dim,
            hid_dims    = hidden_dims,
            out_dim     = out_dim,
            act         = act,
            dropout     = 0.3,
            batch_norm  = batch_norm,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.predictor(h)
    
    def forward(self, h: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h.get_device() != labels.get_device():
            labels = labels.to(h.get_device())

        pred = self.predictor(h)

        loss = self.loss_fn(pred, labels) if self.loss_fn is not None else None
        return loss, pred

