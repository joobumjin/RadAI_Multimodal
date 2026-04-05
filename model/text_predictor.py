from typing import Optional, Tuple, Union
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F
from model.modules import *
from model.losses import *

class EmbPred(nn.Module):
    def __init__(self, 
                 embed_dim: int = 512,
                 predictor_layers: int = 3,
                 hidden_dim: int = 128,
                 out_dim: int = 1,
                 loss_fn: Optional[Callable] = F.mse_loss,
    ):
        super().__init__()
        self.loss_fn = loss_fn

        self.predictor = ProjectionHead(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=out_dim,
            n_layers=predictor_layers
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


class EmbProjector(nn.Module):
    def __init__(self, 
                 embed_dim: int = 512,
                 projector_layers: int = 3,
                 projector_hidden: int = 384,
                 projected_dim: int = 256,
                 predictor_hidden: list[int] = [128],
                 ssl_loss_fn: Callable = InfoNCELoss(),
                 class_loss_fn: Callable = F.mse_loss,
    ):
        super().__init__()
        self.ssl_loss_fn = ssl_loss_fn
        self.class_loss_fn = class_loss_fn

        self.projector = ProjectionHead(
            input_dim=embed_dim,
            hidden_dim=projector_hidden,
            output_dim=projected_dim,
            n_layers=projector_layers
        )

        self.predictor = create_mlp(
            in_dim = projected_dim,
            hid_dims = predictor_hidden,
            out_dim = 1,
            end_with_fc=True
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
    
    def project(self, view1: torch.Tensor, view2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        view1_proj = self.projector(view1)
        view2_proj = self.projector(view2)

        loss = self.ssl_loss_fn(view1_proj, view2_proj)

        return loss, view1_proj, view2_proj
    
    def classify(self, h: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h.get_device() != labels.get_device():
            labels = labels.to(h.get_device())
        
        proj = self.projector(h)

        pred = self.predictor(proj)

        loss = self.class_loss_fn(pred, labels)
        return loss, pred