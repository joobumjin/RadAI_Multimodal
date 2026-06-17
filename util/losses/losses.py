from torch.nn import functional as F
from .cox_loss import *

LOSS_FNS = {
    "l1": lambda preds, x: F.l1_loss(preds, x["label"]),
    "smooth_l1": lambda preds, x: F.smooth_l1_loss(preds, x["label"]),
    "mse": lambda preds, x: F.mse_loss(preds, x["label"]),
    "bce": lambda preds, x: F.binary_cross_entropy_with_logits(preds, x["label"]),
    "cox_nll": cox_nll_loss
}