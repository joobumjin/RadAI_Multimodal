import numpy as np
import torch

from sksurv.metrics import concordance_index_censored


def calculate_c_indices(model: torch.nn.Module, train_loader, test_loader, device):
    model.eval()

    train_preds, train_deaths, train_times = [], [], []
    test_preds, test_deaths, test_times = [], [], []

    for batch in train_loader:
        vitals = batch["vital_status"].numpy().astype(bool)
        times = batch["survival_months"].numpy()
        train_deaths.append(~vitals)
        train_times.append(times)

        for key in batch: batch[key] = batch[key].to(device)
        with torch.inference_mode():
            preds = model.predict(batch)

            preds = preds.detach().squeeze(-1).cpu().numpy()
            train_preds.append(preds)
    
    train = [np.concatenate(l) for l in [train_deaths, train_times, train_preds]]

    for batch in test_loader:
        vitals = batch["vital_status"].numpy().astype(bool)
        times = batch["survival_months"].numpy()
        test_deaths.append(~vitals)
        test_times.append(times)

        for key in batch: batch[key] = batch[key].to(device)
        with torch.inference_mode():
            preds = model.predict(batch)

            preds = preds.detach().squeeze(-1).cpu().numpy()
            test_preds.append(preds)

    test = [np.concatenate(l) for l in [test_deaths, test_times, test_preds]]

    train_c, _, _, _, _ = concordance_index_censored(*train)
    test_c, _, _, _, _ = concordance_index_censored(*test)

    combined = [np.concatenate(arrs) for arrs in zip(train, test)]

    combined_c, _, _, _, _ = concordance_index_censored(*combined)

    return {"Train C-Index": train_c, "Test C-Index": test_c, "Combined C-Index": combined_c}