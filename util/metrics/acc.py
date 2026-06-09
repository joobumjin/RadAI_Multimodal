import numpy as np
import torch

def compile_survival(model, loader, device):
    preds, ids, gt = [], [], []
    for batch in loader:
        ids.append(batch["slide_ids"].numpy().squeeze(-1))
        gt.append(batch["survival_days"].numpy().squeeze(-1) > 730.0)

        for key in batch: batch[key] = batch[key].to(device)
        with torch.inference_mode():
            p = model.predict(batch)

            #in this case, we have survival probs 
            p = (torch.sigmoid(p.detach().squeeze(-1)) > 0.5).numpy().squeeze(-1)
            preds.append(p)
    
    return [np.concatenate(l) for l in [ids, gt, preds]]

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

def confusion_matrix(conf_counts, conf_names):
    import plotly.express as px


    fig = px.imshow(conf_counts, 
                    labels=dict(x="Predicted", y="Ground Truth", color="Count"), 
                    x=['True', 'False'], 
                    y=['True', 'False'], 
                    text_auto=True,
                    custom_data=[conf_names])
    fig.update_traces(
        hovertemplate="{customdata[0]}"
    )

    return fig

def get_tp_fp(model: torch.nn.Module, train_loader, val_loader, test_loader, device, compile = None):
    def tp_fp(ids, gt, preds):
        pos_mask = gt == 1.
        neg_mask = ~pos_mask

        tp_mask = preds[pos_mask] == 1.
        fp_mask = preds[pos_mask] == 0.

        tn_mask = preds[neg_mask] == 0.
        fn_mask = preds[neg_mask] == 1.

        # counts = {
        #     "true pos": np.sum(tp_mask),
        #     "false pos": np.sum(fp_mask),
        #     "true neg": np.sum(tn_mask),
        #     "false neg": np.sum(fn_mask),
        # }

        counts = [[np.sum(tp_mask), np.sum(fp_mask)], 
                  [np.sum(tn_mask), np.sum(fn_mask)]]

        # id_split =  {
        #     "true pos": ids[tp_mask],
        #     "false pos": ids[fp_mask],
        #     "true neg": ids[tn_mask],
        #     "false neg": ids[fn_mask],
        # }

        id_split = [[ids[tp_mask], ids[fp_mask]],
                    [ids[tn_mask], ids[fn_mask]]]

        return counts, id_split

    compile_fn = compile if compile is not None else compile_survival
    mats = {}

    train = compile_fn(model, train_loader, device)
    train_stats = tp_fp(*train)
    mats["Train Confusion Matrix"] = confusion_matrix(*train_stats)

    if val_loader is not None:        
        val = compile_fn(model, val_loader, device)
        val_stats = tp_fp(*val)
        mats["Validation Confusion Matrix"] = confusion_matrix(*val_stats)

    test = compile_fn(model, test_loader, device)
    test_stats = tp_fp(*test)
    mats["Test Confusion Matrix"] = confusion_matrix(*test_stats)

    return mats

    

    