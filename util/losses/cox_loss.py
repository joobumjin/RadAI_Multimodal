from torch import Tensor
from torchsurv.loss.cox import *

def cox_nll_loss(preds: Tensor, x: dict[str, Tensor]):
    """
    Cox Negative Log Likelihood Loss

    Parameters
    ----------
    preds : torch.Tensor
        Log hazards 
    x : dict[str, Tensor]
        Batch dict with keys
        - "survival_right_censor": per patient surviving or not
        - "survival_days": survival in days

    Returns
    ----------
    loss : torch.Tensor
        Cox NLL Loss value
    """
    surviving = ~(x["survival_right_censor"].squeeze(-1).bool())
    time = x["survival_days"].squeeze(-1)

    return neg_partial_log_likelihood(log_hz=preds, event=surviving, time=time)