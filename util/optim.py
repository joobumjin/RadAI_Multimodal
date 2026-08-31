from typing import Optional
from argparse import Namespace

from torch import nn
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def get_opt_and_sched(model: nn.Module, args: Namespace, iter_per_epoch: Optional[int] = None):
    """
    returns optimizer and linear warm up + cosine annealed LR scheduler

    params:
    - model: model to optimize
    - args: argparse.Namespace
        - warmup_epochs: int, num warmup epochs
        - epochs: total epochs to train for (including warm up)
        - min_lr: minimum lr to decay to
    - iter_per_epoch: number of scheduler steps per epoch (if stepping for every train step instead of once an epoch)
    """

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    #linear warmup
    wu_iters = args.warmup_epochs * iter_per_epoch if iter_per_epoch is not None else args.warmup_epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=args.warmup_start, end_factor=1.0, total_iters=wu_iters)

    #cosine anneal 
    t_max = (args.epochs - args.warmup_epochs) * iter_per_epoch if iter_per_epoch is not None else (args.epochs - args.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.min_lr)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[wu_iters])

    return optimizer, scheduler