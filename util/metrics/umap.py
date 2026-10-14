from collections import defaultdict
from argparse import Namespace

import seaborn as sns

import torch 
from torch.utils.data import DataLoader
import umap
from ..consts import *

def compile_logits(model, device, loaders: list[DataLoader], args: Namespace):
    model.eval()
    logits = defaultdict(lambda: None)
    labels = None

    for loader in loaders:
        for batch in loader:
            surviving = (batch["survival_right_censor"].squeeze(-1).bool()) #true if could have lived longer
            times = batch["survival_days"].squeeze(-1)

            time = times > args.survival_years * DAYS_IN_YR
            event = ~surviving
            label = (event * 2 + time)
            labels = label.detach().cpu() if labels is None else torch.cat([labels, label.detach().cpu()])
    
            for key in batch: batch[key] = batch[key].to(device)

            with torch.inference_mode():
                _, _, logit = model.predict(batch, True, True)
            logits = {key: (arr.detach().cpu() if logits[key] is None 
                            else torch.cat([logits[key], arr.detach().cpu()])) for key, arr in logit.items()}

    return logits, labels

def plot_umap(logit: torch.Tensor, labels: torch.Tensor, model_name: str):
    sns.set_theme()

    logit = logit.numpy()

    label_text = {0: "Live Uncens.", 1: "Dead Uncens.", 2: "Live Cens.", 3: "Dead Uncens."}
    labels = [label_text[l.item()] for l in labels]
            
    mapper = umap.UMAP().fit(logit)
    ax = sns.scatter(x=mapper.embedding_[:, 0], y=mapper.embedding_[:, 1], hue=labels)
    fig = ax.get_figure()
    fig.savefig(f"vises/umap/{model_name}.pdf", dpi=300)

    del mapper
