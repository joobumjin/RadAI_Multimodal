from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def get_opt_and_sched(model, args, iter_per_epoch = None):
    optimizer = optim.AdamW([{"params": enc.parameters(), "lr": args.lr} for enc in model.encoders.values()], 
                            lr=args.lr, 
                            betas=(0.9, 0.95))

    #linear warmup
    wu_iters = args.warmup_epochs * iter_per_epoch if iter_per_epoch is not None else args.warmup_epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=args.warmup_start, end_factor=1.0, total_iters=wu_iters)
    #cosine anneal 
    t_max = (args.epochs - args.warmup_epochs) * iter_per_epoch if iter_per_epoch is not None else (args.epochs - args.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.min_lr)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[wu_iters])

    return optimizer, scheduler