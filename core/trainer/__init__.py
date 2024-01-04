from .trainer import Trainer
from .kd_trainer import KDTrainer
from .imtl_trainer import IMTLTrainer
from .pcgrad_trainer import PCGradTrainer, CAGradTrainer


def build_trainer(args, net, data_loader, teachers=None):
    if args.loss_metric.lower() == 'imtl':
        return IMTLTrainer(args, net, data_loader, teachers)
    if 'pcgrad' in args.loss_metric.lower():
        return PCGradTrainer(args, net, data_loader, teachers)
    if 'cagrad' in args.loss_metric.lower():
        return CAGradTrainer(args, net, data_loader, teachers)
    if args.loss_metric.lower() == 'kd_mtl':
        return KDTrainer(args, net, data_loader, teachers)
    else:
        return Trainer(args, net, data_loader, teachers)
