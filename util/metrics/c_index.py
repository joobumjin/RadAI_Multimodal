import numpy as np
import torch

from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

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
            preds = model.predict(batch).detach().squeeze(-1).cpu().numpy() #predicted hazards
            split_preds.append(preds)
    
    return [np.concatenate(l) for l in [split_deaths, split_times, split_preds]]


def calculate_c_indices_auc(model: torch.nn.Module, train_loader, val_loader, test_loader, device, surv_yr=2.0, compile = None):
    """
    Required Loader Batch Keys:
    - "survival_right_censor": Right censor on event date
    - "survival_days"
    - Whatever else the model needs
    """
    compile_fn = compile if compile is not None else compile_split    

    train = compile_fn(model, train_loader, device)
    num_train = len(train[0])

    if val_loader is not None:        
        val = compile_fn(model, val_loader, device)
        num_val = len(val[0])
    else:
        val = []
        num_val = 0

    test = compile_fn(model, test_loader, device)
    num_test = len(test[0])

    train_c, _, _, _, _ = concordance_index_censored(*train)
    val_c = concordance_index_censored(*val)[0] if len(val) else None
    test_c, _, _, _, _ = concordance_index_censored(*test)

    surv_train  = np.zeros(num_train,   dtype=[('event', bool), ('surv_time', float)])
    surv_val    = np.zeros(num_val,     dtype=[('event', bool), ('surv_time', float)])
    surv_test   = np.zeros(num_test,    dtype=[('event', bool), ('surv_time', float)])

    surv_train['event'] = train[0]
    surv_train['surv_time'] = train[1]

    if num_val > 0:
        surv_val['event'] = val[0]
        surv_val['surv_time'] = val[1]

    surv_tv = np.concatenate((surv_train, surv_val))

    surv_test['event'] = test[0]
    surv_test['surv_time'] = test[1]

    train_auc, _    = cumulative_dynamic_auc(surv_tv,   surv_train, train[2],   [surv_yr * 365.0], tied_tol=1e-08)
    val_auc         = cumulative_dynamic_auc(surv_tv,   surv_val,   val[2],     [surv_yr * 365.0], tied_tol=1e-08)[0] if len(val) else None
    test_auc, _     = cumulative_dynamic_auc(surv_test, surv_test,  test[2],    [surv_yr * 365.0], tied_tol=1e-08)

    ret = {
        "Train C-Index": train_c, 
        "Test C-Index": test_c, 
        "Train AUC": train_auc[0], 
        "Test AUC": test_auc[0], 
    }

    if len(val): 
        ret["Validation C-Index"] = val_c
        ret["Validation AUC"] = val_auc[0]

    return ret