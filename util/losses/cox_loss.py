from torchsurv.loss.cox import *

def cox_nll_loss(preds, x):
    """
    Cox Negative Log Likelihood Loss
    Params:
    - preds: Log hazards 
    - x: Batch dict with keys
        - "survival_right_censor": per patient surviving or not
        - "survival_days": survival in days
    """
    surviving = x["survival_right_censor"].squeeze(-1).bool()
    time = x["survival_days"].squeeze(-1)

    return neg_partial_log_likelihood(log_hz=preds, event=surviving, time=time)