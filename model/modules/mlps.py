import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.seq = seq
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        return self.seq(x)
    
    def predict(self, x):
        return self.seq(x)

def create_mlp(
        in_dim=768, 
        hid_dims=[512, 512], 
        out_dim=512, 
        act=nn.ReLU(),
        dropout=0.,
        end_with_fc=True, 
        end_with_dropout=False,
        bias=True,
        batch_norm=False,
        layer_norm=False,
        rms_norm=False,
        end_with_norm=False,
    ):

    layers = []
    
    for hid_dim in hid_dims:
        layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
        if batch_norm: layers.append(nn.BatchNorm1d(hid_dim))
        elif layer_norm: layers.append(nn.LayerNorm(hid_dim))
        elif rms_norm: layers.append(nn.RMSNorm(hid_dim))
        layers.append(act)
        layers.append(nn.Dropout(dropout))
        in_dim = hid_dim

    layers.append(nn.Linear(in_dim, out_dim))

    if not end_with_fc:
        layers.append(act)

    if end_with_norm:
        if batch_norm: layers.append(nn.BatchNorm1d(out_dim))
        elif layer_norm: layers.append(nn.LayerNorm(out_dim))
        elif rms_norm: layers.append(nn.RMSNorm(out_dim))
    
    if end_with_dropout:
        layers.append(nn.Dropout(dropout))

    mlp = MLP(nn.Sequential(*layers))

    return mlp


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)