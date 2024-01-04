from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR

from .lr_scheduler import PolyDecay
from .warmup_scheduler import GradualWarmupScheduler


def build_scheduler(args, optimizer):
    if args.scheduler.lower() == 'plateau':
        print('ReduceLROnPlateau LR scheduler has been initialized with patience=%d' % args.patience)
        after_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor,
                                            patience=args.patience, min_lr=1e-5)
    elif args.scheduler.lower() == 'poly':
        print('Poly LR scheduler has been initialized')
        after_scheduler = PolyDecay(optimizer, 0.9, args.max_epoch)
    elif args.scheduler.lower() == 'cosine':
        print('Cosine Annealing LR scheduler has been initialized')
        after_scheduler = CosineAnnealingLR(optimizer, args.max_epoch)
    else:
        print('Multi-Step LR scheduler has been initialized')
        after_scheduler = MultiStepLR(optimizer, [args.cold_epoch, args.freeze_epoch], gamma=0.1)

    scheduler = GradualWarmupScheduler(optimizer, 0, args.warmup_epoch, after_scheduler=after_scheduler)
    return scheduler
