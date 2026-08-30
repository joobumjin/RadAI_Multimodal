import torch
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from util.consts import *

def compile_split(model, loader, device):
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

def cind(death, times, preds):
    death = death.bool()

    cindex = ConcordanceIndex()
    c = cindex(preds, death, times)
    return c

def auc(death, times, preds, yrs):
    death = death.bool()

    auc = Auc()
    a = auc(preds, death, times, new_time = torch.ones((1,)) * yrs * DAYS_IN_YR)[0]
    return a


def calculate_c_indices_auc(model: torch.nn.Module, train_loader, val_loader, test_loader, device, surv_yr=2.0, compile_fn = compile_split):
    """
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
    train_a = auc(train_preds, train_deaths, train_times, surv_yr)

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