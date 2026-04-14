from typing import Optional, Tuple, Union
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F
from model.modules import *
from model.losses import *

class EmbMIL(nn.Module):
    def __init__(self, 
                 in_dim: int = 1024,
                 embed_dim: int = 512,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 attn_dim: int = 384,
                 gate: int = True,
                 proj_hidden: int = 128,
                 proj_dim: int = 1,
                 loss_fn: Optional[Callable] = F.mse_loss,
    ):
        super().__init__()
        #all samples in a bag are projected into hidden_sz space
        #and then those projected embs are weighted summed into
        #a single hidden_sz embedding 
        self.loss_fn = loss_fn

        self.down_sampler = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
            act=nn.LeakyReLU(),
            layer_norm=True
        )

        attn_class = GlobalAttention
        self.attn = attn_class(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )

        self.projector = ProjectionHead(
            input_dim=embed_dim,
            hidden_dim=proj_hidden,
            output_dim=proj_dim
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

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only=True) -> torch.Tensor:
        """
        Compute the attention scores (and optionally the embedded features) for the input instances.

        Args:
            h (torch.Tensor): Input tensor of shape [B, M, D], where B is the batch size,
                M is the number of instances (patches), and D is the input feature dimension.
            attn_mask (torch.Tensor, optional): Optional attention mask of shape [B, M], where 1 indicates
                valid positions and 0 indicates masked positions. If provided, masked positions are set to
                a very large negative value before softmax.
            attn_only (bool, optional): If True, return only the attention scores (A).
                If False, return a tuple (h, A) where h is the embedded features and A is the attention scores.

        Returns:
            torch.Tensor: If attn_only is True, returns the attention scores tensor of shape [B, K, M],
                where K is the number of attention heads (usually 1). If attn_only is False, returns a tuple
                (h, A) where h is the embedded features of shape [B, M, D'] and A is the attention scores.
        """
        h = self.down_sampler(h)
        A = self.attn(h)  # B x M x K
        A = torch.transpose(A, -2, -1)  # B x K x M
        if attn_mask is not None:
            # attn_mask: [B, M] with 1 valid, 0 masked
            A = A.masked_fill((1 - attn_mask).unsqueeze(1).bool(), float('-inf'))
        return A if attn_only else (h, A)

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True) -> torch.Tensor:
        """
        Compute bag-level features using attention pooling.

        Args:
            h (torch.Tensor): [B, M, D] input features.
            attn_mask (torch.Tensor, optional): Attention mask.

        Returns:
            Tuple[torch.Tensor, dict]: Bag features [B, D] and attention weights.
        """
        h, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)  # A == B x K x M
        A = F.softmax(A_base, dim=-1)  # softmax over N
        bag = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C
        log_dict = {'attention': A_base.squeeze(1) if return_attention else None}
        return bag, log_dict
    
    def forward(self, h: torch.Tensor, labels: torch.Tensor, attn_mask=None, return_attention: bool = False) -> torch.Tensor:
        if h.get_device() != labels.get_device():
            labels = labels.to(h.get_device())
        
        attn_out, attn_dict = self.forward_features(h, attn_mask=attn_mask,return_attention=return_attention)
        attn_dict = [attn_dict] if return_attention else []

        pred = self.projector(attn_out)
        if self.loss_fn is None:
            return pred, *attn_dict
        
        loss = self.loss_fn(pred, labels)
        return loss, pred, *attn_dict
    
    def predict(self, h: torch.Tensor, attn_mask=None, return_attention: bool = False) -> torch.Tensor:
        attn_out, attn_dict = self.forward_features(h, attn_mask=attn_mask,return_attention=return_attention)
        attn_dict = [attn_dict] if return_attention else []

        pred = self.projector(attn_out)
        return pred, *attn_dict


class SSLEmbMIL(nn.Module):
    def __init__(self, 
                 in_dim: int = 1024,
                 embed_dim: int = 512,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 attn_dim: int = 384,
                 gate: int = True,
                 proj_hidden: int = 384,
                 proj_dim: int = 256,
                 num_predictor_layers: int = 1,
                 pred_dim: int = 1,
                 pred_hidden: int = 128,
                 ssl_loss_fn: Callable = InfoNCELoss(),
                 class_loss_fn: Callable = F.mse_loss,
    ):
        super().__init__()
        self.ssl_loss_fn = ssl_loss_fn
        self.class_loss_fn = class_loss_fn

        self.projector = EmbMIL(
            in_dim,
            embed_dim,
            num_fc_layers,
            dropout,
            attn_dim,
            gate,
            proj_hidden,
            proj_dim,
            loss_fn = None
        )

        self.predictor = create_mlp(
            in_dim=proj_dim,
            hid_dims=[pred_hidden] * (num_predictor_layers - 1),
            dropout=dropout,
            out_dim=pred_dim,
            end_with_fc=False
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
    
    def project_batch(self, 
                      view1: torch.Tensor, view2: torch.Tensor, 
                      mask1: torch.Tensor, mask2: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        view1_proj = self.projector.predict(view1, attn_mask=mask1, return_attention=False)[0]
        view2_proj = self.projector.predict(view2, attn_mask=mask2, return_attention=False)[0]

        loss = self.ssl_loss_fn(view1_proj, view2_proj)

        return loss, view1_proj, view2_proj
    
    def classify(self, h: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h.get_device() != labels.get_device():
            labels = labels.to(h.get_device())
        
        proj = self.projector.predict(h, return_attention=False)[0]

        pred = self.predictor(proj)

        loss = self.class_loss_fn(pred, labels)
        return loss, pred
