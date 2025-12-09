import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    def __init__(self, emb_sz: int = 128, hidden_sz: int = 128, branches: int = 1):
        super().__init__()
        #learn linear weights for each instance in a MIL bag for attn
        self.emb_sz = emb_sz
        self.hidden_sz = hidden_sz
        self.branches = branches
        
        self.attention = nn.Sequential(
            nn.Linear(self.emb_sz, self.hidden_sz),
            nn.Tanh(),
            nn.Linear(self.hidden_sz, self.branches),
        )

    """
    Creates branches # of attention embeddings 
    params:
    - x: embeddings (..., num_instances, emb_sz)
    returns:
    - y: (hidden_sz * branches, )
    """
    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim = 0) # num_instances, branches

        dims = list(range(x.dim()))
        dims[-1], dims[-2] = dims[-2], dims[-1] # swap last two dims of x

        return torch.flatten(x.permute(dims) @ attn_weights, start_dim=-2, end_dim=-1) # (..., branches * emb_sz,)

