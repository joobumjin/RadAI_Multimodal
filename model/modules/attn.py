import torch
from torch.nn import functional as F
from torch import nn


class GlobalAttention(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        num_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., num_classes=1):
        super().__init__()

        self.attn = nn.Sequential(nn.Linear(L, D),
                                  nn.Tanh(),
                                  nn.Dropout(dropout),
                                  nn.Linear(D, num_classes))

    def forward(self, x):
        return self.attn(x)  # N x num_classes

class GlobalGatedAttention(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        num_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., num_classes=1):
        super().__init__()

        self.attn_a = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)
        ]

        self.attn_b = [
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        ]

        self.attn_a = nn.Sequential(*self.attn_a)
        self.attn_b = nn.Sequential(*self.attn_b)
        self.attn_c = nn.Linear(D, num_classes)

    def forward(self, x):
        a = self.attn_a(x)
        b = self.attn_b(x)
        A = a.mul(b)
        A = self.attn_c(A)  # N x num_classes
        return A
