import numpy as np
import torch

from sksurv.metrics import concordance_index_censored

def compile_split(model, loader, device):
    model.eval()

    split_preds, split_deaths, split_times = [], [], []
    for batch in loader:
        surviving = (batch["survival_right_censor"].numpy().squeeze(-1).astype(bool)) #true if died
        times = batch["survival_days"].numpy().squeeze(-1)
        split_deaths.append(~surviving)
        split_times.append(times)

        for key in batch: batch[key] = batch[key].to(device)
        with torch.inference_mode():
            preds = model.predict(batch)

            #convert to risk score, in this case the conversion doesnt matter as long as the relative risk values make sense (higher risk = lower survival likelihood)
            preds = 1 - preds.detach().squeeze(-1).cpu().numpy()
            split_preds.append(preds)
    
    return [np.concatenate(l) for l in [split_deaths, split_times, split_preds]]


def calculate_c_indices(model: torch.nn.Module, train_loader, val_loader, test_loader, device, compile = None):
    """
    Required Loader Batch Keys:
    - "survival_right_censor": Right censor on event date
    - "survival_days"
    - Whatever else the model needs
    """
    compile_fn = compile if compile is not None else compile_split
    
    train = compile_fn(model, train_loader, device)

    if val_loader is not None:        
        val = compile_fn(model, val_loader, device)
    else:
        val = []

    test = compile_fn(model, test_loader, device)

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