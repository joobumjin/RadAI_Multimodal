import numpy as np
import torch

from sksurv.metrics import concordance_index_censored

def compile_split(model, loader, device):
    split_preds, split_deaths, split_times = [], [], []
    for batch in loader:
        surviving = batch["survival_right_censor"].numpy().astype(bool)
        times = batch["survival_days"].numpy()
        split_deaths.append(~surviving)
        split_times.append(times)

        for key in batch: batch[key] = batch[key].to(device)
        with torch.inference_mode():
            preds = model.predict(batch)

            preds = preds.detach().squeeze(-1).cpu().numpy()
            split_preds.append(preds)
    
    return [np.concatenate(l) for l in [split_deaths, split_times, split_preds]]


def calculate_c_indices(model: torch.nn.Module, train_loader, val_loader, test_loader, device):
    """
    Required Loader Batch Keys:
    - "survival_right_censor": Right censor on event date
    - "survival_days"
    - Whatever else the model needs
    """

    model.eval()
  
    train = compile_split(model, train_loader, device)

    if val_loader is not None:        
        val = compile_split(model, val_loader, device)
    else:
        val = []

    test = compile_split(model, test_loader, device)

    train_c, _, _, _, _ = concordance_index_censored(*train)
    val_c = concordance_index_censored(*val)[0] if len(val) else None
    test_c, _, _, _, _ = concordance_index_censored(*test)

   
    if len(val): data = [train, val, test]
    else: data = [train, test]

    combined = [np.concatenate(arrs) for arrs in zip(*data)]
    combined_c, _, _, _, _ = concordance_index_censored(*combined)

    ret = {
        "Train C-Index": train_c, 
        "Test C-Index": test_c, 
        "Combined C-Index": combined_c
    }

    if len(val): ret["Validation C-Index"] = val_c

    return ret