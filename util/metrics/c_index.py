import numpy as np
import torch

from sksurv.metrics import concordance_index_censored


def calculate_c_indices(model: torch.nn.Module, train_loader, val_loader, test_loader, device):
    model.eval()

    train_preds, train_deaths, train_times = [], [], []
    val_preds, val_deaths, val_times = [], [], []
    test_preds, test_deaths, test_times = [], [], []

    for batch in train_loader:
        surviving = batch["survival_right_censor"].numpy().astype(bool)
        times = batch["survival_days"].numpy()
        train_deaths.append(~surviving)
        train_times.append(times)

        for key in batch: batch[key] = batch[key].to(device)
        with torch.inference_mode():
            preds = model.predict(batch)

            preds = preds.detach().squeeze(-1).cpu().numpy()
            train_preds.append(preds)
    
    train = [np.concatenate(l) for l in [train_deaths, train_times, train_preds]]

    for batch in val_loader:
        surviving = batch["survival_right_censor"].numpy().astype(bool)
        times = batch["survival_days"].numpy()
        val_deaths.append(~surviving)
        val_times.append(times)

        for key in batch: batch[key] = batch[key].to(device)
        with torch.inference_mode():
            preds = model.predict(batch)

            preds = preds.detach().squeeze(-1).cpu().numpy()
            val_preds.append(preds)
    
    val = [np.concatenate(l) for l in [val_deaths, val_times, val_preds]]

    for batch in test_loader:
        surviving = batch["survival_right_censor"].numpy().astype(bool)
        times = batch["survival_days"].numpy()
        test_deaths.append(~surviving)
        test_times.append(times)

        for key in batch: batch[key] = batch[key].to(device)
        with torch.inference_mode():
            preds = model.predict(batch)

            preds = preds.detach().squeeze(-1).cpu().numpy()
            test_preds.append(preds)

    test = [np.concatenate(l) for l in [test_deaths, test_times, test_preds]]

    train_c, _, _, _, _ = concordance_index_censored(*train)
    val_c, _, _, _, _ = concordance_index_censored(*val)
    test_c, _, _, _, _ = concordance_index_censored(*test)

    combined = [np.concatenate(arrs) for arrs in zip(train, val, test)]

    combined_c, _, _, _, _ = concordance_index_censored(*combined)

    return {"Train C-Index": train_c, "Validation C-Index": val_c, "Test C-Index": test_c, "Combined C-Index": combined_c}