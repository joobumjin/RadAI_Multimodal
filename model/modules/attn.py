from torch import Tensor, nn

class GlobalAttention(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        num_classes: number of classes
    """

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0., num_classes: int = 1):
        super().__init__()

        self.attn = nn.Sequential(nn.Linear(L, D),
                                  nn.Tanh(),
                                  nn.Dropout(dropout),
                                  nn.Linear(D, num_classes))

    def forward(self, x: Tensor) -> Tensor:
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

    def __init__(self, L:int = 1024, D: int = 256, dropout: float = 0., num_classes: int = 1):
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

    def forward(self, x: Tensor) -> Tensor:
        a = self.attn_a(x)
        b = self.attn_b(x)
        A = a.mul(b)
        A = self.attn_c(A)  # N x num_classes
        return A
