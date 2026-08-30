import torch
import torch.nn.functional as F

class InfoNCELoss(torch.nn.Module):
    """
    InfoNCE Loss On Paired Samples
    Assumes that z1 and z2 are encoded versions of 2 different views of the same samples.
    """
    def __init__(self, temperature=0.3):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Calculates InfoNCE loss with identity samples being positive pairs
        and all others being negative pairs.
        params:
        - z1: encoded view 1 [B, D]
        - z2: encoded view 2 [B, D]
        """
        B = len(z1)
        
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        z = torch.cat([z1, z2], dim=0)
        sim = (z @ z.t()) / self.temperature
        
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))
        
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),    #match z1 to z2
            torch.arange(0, B, device=z.device),        #match z2 to z1
        ])
        
        return F.cross_entropy(sim, labels)
