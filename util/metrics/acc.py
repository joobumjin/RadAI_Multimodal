import torch

def cat_acc(preds: torch.Tensor, labels: torch.Tensor):
    """
    Calculates Average Categorical Accuracy
    
    :param preds: long tensor [N, C], 
    :param labels: long Tensor [N] with values [0, C-1]
    """

    return torch.sum(preds.detach().argmax(axis=-1, keepdim=True) == labels.detach()) / len(preds)

def acc(preds: torch.Tensor, labels: torch.Tensor):
    """
    Calculates Average Accuracy
    
    :param preds: long tensor [N], 
    :param labels: long Tensor [N]
    """

    return torch.sum(preds.detach() == labels.detach()) / len(preds)