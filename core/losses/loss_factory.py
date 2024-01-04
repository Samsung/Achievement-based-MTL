from functools import partial
from .multi_task import WeightedSum, Geometric, GradNorm, IMTL, IMTLG, DTP, MTLwFW, AMTLG, AMTLA, DWA, MGDA, RLW
from .multi_task import DWAG, RLWG, DTPG  # Modified for geometric mean

loss_metric_list = [
    'weighted-sum',
    'geometric',
    'dwa', 'rlw',  # base on loss magnitude
    'grad-norm', 'imtl-g', 'imtl', 'mgda',  # based on gradient
    'dtp', 'fmtl', 'amtl', 'amtl-a',  # based on validation accuracy
    'pcgrad', 'pcgrad-amtl', 'cagrad', 'cagrad-amtl',  # based on optimization. independent to multi-task loss
    'kd-mtl', 'kd-amtl',  # based on knowledge distillation. independent to multi-task loss
    'dwa-g', 'rlw-g', 'dtp-g'  # Modified for geometric mean
]


def loss_factory(args, net):
    loss_metric = args.loss_metric.lower()
    last_shared_layer = net.get_last_shared_params()

    loss_dict = {
        'weighted-sum': partial(WeightedSum),
        'dwa': partial(DWA),
        'dwa-g': partial(DWAG),
        'rlw': partial(RLW),
        'rlw-g': partial(RLWG),
        'geometric': partial(Geometric),
        'grad-norm': partial(GradNorm, lr=args.lr, shared_weights=last_shared_layer),
        'imtl': partial(IMTL, shared_weights=last_shared_layer),
        'imtl-g': partial(IMTLG, shared_weights=last_shared_layer),
        'mgda': partial(MGDA, shared_weights=net.get_shared_params()),
        'dtp': partial(DTP, focusing_factor=args.focusing_factor),
        'dtp-g': partial(DTPG, focusing_factor=args.focusing_factor),
        'fmtl': partial(MTLwFW, focusing_factor=args.focusing_factor),
        'amtl': partial(AMTLG, potential=args.potential, focusing_factor=args.focusing_factor, margin=args.margin),
        'amtl-a': partial(AMTLA, potential=args.potential, focusing_factor=args.focusing_factor, margin=args.margin),
        'pcgrad': partial(WeightedSum),
        'cagrad': partial(WeightedSum),
        'pcgrad-amtl': partial(AMTLA, potential=args.potential, focusing_factor=args.focusing_factor, margin=args.margin),
        'cagrad-amtl': partial(AMTLA, potential=args.potential, focusing_factor=args.focusing_factor, margin=args.margin),
        'kd-mtl': partial(WeightedSum),
        'kd-amtl': partial(AMTLG, potential=args.potential, focusing_factor=args.focusing_factor, margin=args.margin),
    }
    assert loss_metric in loss_dict, f'{loss_metric} is not supported for loss metric'
    loss_function = loss_dict[loss_metric](args.app, args.loss_weight)

    if 'detection' in args.app:
        loss_function.register_loss('detection', (args.cls_loss_type, args.loc_loss_type),
                                    prior=net.get_priors(), iou_aware=args.iou_aware_cls, mix_up=args.mix_up,
                                    label_smoothing=args.label_smoothing, loc_weight=args.loc_weight)
    if 'segmentation' in args.app:
        loss_function.register_loss('segmentation', 'cross_entropy', smoothing=args.label_smoothing, ignore_index=255)
    if 'depth' in args.app:
        loss_function.register_loss('depth', 'l1_loss')
    if 'normal' in args.app:
        loss_function.register_loss('normal', 'cosine_loss')

    return loss_function
