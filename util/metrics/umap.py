from collections import defaultdict
from typing import Optional
from argparse import Namespace

import seaborn as sns

import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader
import umap
from ..consts import *

def compile_logits(model: nn.Module, device: torch.device, loaders: list[DataLoader], args: Namespace):
    """
    Compile a model's projected modality logits alongside label values

    Parameters
    ----------
    model : nn.Module
        Model to gather projected embs from
    device : torch.device
        Torch device to run inference on
    loaders : list[DataLoader]
        All dataloaders to compile from
    args : Namespace
        Compiled CLI args

    Returns
    ----------
    logits : dict[str, np.ndarray]
        modality string: (n_sample, emb_size) NumPy Array for all modalities
    labels : np.ndarray
        NumPy Array with all sample labels for UMAP visualization with values 0 to 3, inclusive
    """
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
            labels = label.detach().cpu().numpy() if labels is None else np.concat([labels, label.detach().cpu().numpy()])
    
            for key in batch: batch[key] = batch[key].to(device)

            with torch.inference_mode():
                _, _, logit = model.predict(batch, True, True)
            logits = {key: (arr.detach().cpu().numpy() if logits[key] is None 
                            else torch.cat([logits[key], arr.detach().cpu().numpy()])) for key, arr in logit.items()}

    return logits, labels

def plot_umap(logit: np.ndarray, model_name: str, labels: Optional[np.ndarray] = None):
    """
    Performs dimensionality reduction on a model's projected embeddings with UMAP and plots

    Parameters
    ----------
    logits : np.ndarray
        The projected embeddings for a model's modality
    labels : np.ndarray
        The labels with which to color each point, values 0-3. inclusive
    model_name : string
        Just the name of the model for file saving
    """
    sns.set_theme()

    label_args = {}

    if labels is not None:
        label_text = {0: "Live Uncens.", 1: "Dead Uncens.", 2: "Live Cens.", 3: "Dead Uncens."}
        labels = [label_text[l.item()] for l in labels]
        label_args["hue"] = labels
            
    mapper = umap.UMAP().fit(logit)
    ax = sns.scatter(x=mapper.embedding_[:, 0], y=mapper.embedding_[:, 1], **label_args)
    fig = ax.get_figure()
    fig.savefig(f"vises/umap/{model_name}.pdf", dpi=300)

    del mapper
