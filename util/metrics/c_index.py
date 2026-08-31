import torch
from torch.utils.data import DataLoader
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from util.consts import *

def compile_split(model, loader, device):
    """
    """

    model.eval()

    split_preds, split_deaths, split_times = [], [], []
    for batch in loader:
        surviving = (batch["survival_right_censor"].squeeze(-1).bool()) #true if could have lived longer
        times = batch["survival_days"].squeeze(-1)
        split_deaths.append(~surviving)
        split_times.append(times)

        for key in batch: batch[key] = batch[key].to(device)
        with torch.inference_mode():
            preds = model.predict(batch).detach().squeeze(-1).cpu() #predicted hazards
            split_preds.append(preds)
    
    return [torch.cat(l) for l in [split_deaths, split_times, split_preds]]

def cind(death: torch.Tensor, times: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    """
    Calculates C-Index of preds

    params:
        - deaths: bool tensor indicator for survival event
        - times: float tensor for duration from diagnosis to last follow up
        - preds: float tensor with model risk scores
    """
    death = death.bool()

    cindex = ConcordanceIndex()
    c = cindex(preds, death, times)
    return c

def auc(death: torch.Tensor, times: torch.Tensor, preds: torch.Tensor, yrs: int):
    """
    Calculates Censored AUC of preds

    params:
        - deaths: bool tensor indicator for survival event
        - times: float tensor for duration from diagnosis to last follow up
        - preds: float tensor with model risk scores
        - yrs: int time at which to evaluate AUC
    """

    death = death.bool()

    auc = Auc()
    a = auc(preds, death, times, new_time = torch.tensor(yrs * DAYS_IN_YR))[0]
    return a


def calculate_c_indices_auc(model: torch.nn.Module, device: torch.device, 
                            train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, 
                            surv_yr: int = 2.0, compile_fn: callable = compile_split):
    """
    Calculates the C-Index and AUC for each of 3 splits

    params:
    - model: model to evaluate
    - device: model's device
    - train_loader: data loader for train split
    - val_loader: data loader for validation split
    - test_loader: data loader for test split
    - surv_yr: time at which AUC will be evaluated
    - compile_fn: function used to pull data from the dataloader with the model

    Required Loader Batch Keys:
    - "survival_right_censor": Right censor on event date
    - "survival_days"
    - Whatever else the model needs
    """
    train_deaths, train_times, train_preds = compile_fn(model, train_loader, device)

    if val_loader is not None:
        val_deaths, val_times, val_preds = compile_fn(model, val_loader, device)
        num_val = len(val_deaths)
    else:
        val_deaths, val_times, val_preds = [], [], []
        num_val = 0

    test_deaths, test_times, test_preds = compile_fn(model, test_loader, device)

    train_c = cind(train_deaths, train_times, train_preds)
    train_a = auc(train_deaths, train_times, train_preds, surv_yr)

    if num_val:
        val_c = cind(val_deaths, val_times, val_preds)
        val_a = auc(val_deaths, val_times, val_preds, surv_yr)
    else:
        val_c = None
        val_a = None

    test_c = cind(test_deaths, test_times, test_preds)
    test_a = auc(test_deaths, test_times, test_preds, surv_yr)

    ret = {
        "Train C-Index": train_c, 
        "Test C-Index": test_c, 
        "Train AUC": train_a, 
        "Test AUC": test_a, 
    }

    if num_val: 
        ret["Validation C-Index"] = val_c
        ret["Validation AUC"] = val_a

    return ret