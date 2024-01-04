import os
import sys
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist

from ..losses import loss_metric_list
from ..optimizer import optimizer_list
arch_list = ['VMM', 'EfficientDet', 'DeepLab']
model_list = ['vgg16', 'resnet50', 'resnet101',
              'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_v2_s',
              'mobilenet-v1', 'mobilenet_v2', ]


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip() or (arg.strip() == '#'):
            return
        yield arg


def get_arguments():
    parser = argparse.ArgumentParser(description='Vision Multi-Task Learning', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--app', '--applications', nargs='+', default=['detection', 'segmentation'],
                        help='detection, segmentation, depth')

    parser.add_argument('-a', '--arch', default='VMM', choices=arch_list)
    parser.add_argument('--basenet', default='efficientnet_v2_s', choices=model_list)
    parser.add_argument('--frozen_stages', default=4, type=int,
                        help='Number of the frozen stages of the feature extractor'
                             'Please sef None if you want to train all layers')
    parser.add_argument('--sync_bn', default=False, action='store_true', help='Apply synchronized batchnorm or not')
    parser.add_argument('--normal_type', default='conv', help='Type of normalization layers')
    parser.add_argument('--normal_size', default=64, type=int, help='Number of channels for normal layers')
    parser.add_argument('--head_type', default='mbconv', help='Type of prediction head')
    parser.add_argument('--activation', default='relu6', help='Type of activations')
    parser.add_argument('--cls_head', default='softmax', help='Type of classification head for detection')
    parser.add_argument('--use_FPN', default='triv2-bifpn', type=str, help='Type of FPN modules')
    parser.add_argument('--output_stride', default=16, type=int, help='Stride of segmentation output')
    parser.add_argument('--skip_connection', default=0, type=int,
                        help='DeepLab uses skip connection if it is positive number'
                             'The positive number denotes the number of channels for skip connection')
    parser.add_argument('--shared_head', default=True, type=str2bool,
                        help='DeepLab shares the ASPP module among tasks if it is True')
    parser.add_argument('--from_scratch', default=False, type=str2bool,
                        help='Use or do not use pretrained weight of base network')

    parser.add_argument('-s', '--size', nargs='+', default=[640, 640], type=int,
                        help='If got two inputs (height, width), input image would be rectangular')
    parser.add_argument('--training_size', nargs='+', default=[], type=int,
                        help='Use only if image sizes for training and evaluation are different')

    parser.add_argument('--train_dataset', nargs='+', default=['VOC'], help='VOC, COCO, or NYU dataset')
    parser.add_argument('--test_dataset', nargs='+', default=['VOC'], help='VOC, COCO, or NYU dataset')
    parser.add_argument('--year', default='2007', help='2007 or 2012, Year for the PASCAL VOC dataset')
    parser.add_argument('--dataset_root', required=True)

    parser.add_argument('--min_scale', default=2/3, type=float, help='Minimum scale for RandScale Transform')
    parser.add_argument('--max_scale', default=4/3, type=float, help='Maximum scale for RandScale Transform')

    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--optim', default='sgd', type=str, help='Optimizer', choices=optimizer_list)
    parser.add_argument('--num_workers', default=16, type=int, help='Number of workers used in data loading')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for optimizer')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for optimizer')

    parser.add_argument('--tencent_trick', default=False, action='store_true', help='Set true if uses tencent_trick')
    parser.add_argument('--multiple_lr', default=1, type=int, help="Set 1 if don't use multiple lr")

    parser.add_argument('--model-ema', default=False, action='store_true', help='Set true if uses model ema')
    parser.add_argument('--model-ema-momentum', default=0.9999, type=float, help='Momentum of model ema')

    parser.add_argument('--cls_loss_type', default='CE', help='Classification loss type for detection app',
                        choices=['CE', 'focal'])
    parser.add_argument('--loc_loss_type', default='IoU', help='Regression loss type for detection app',
                        choices=['L1', 'IoU'])
    parser.add_argument('--iou_aware_cls', default=False, help='Set True if use IoU-aware classification loss',
                        action='store_true')
    parser.add_argument('--loc_weight', default=10.0, type=float, help='Weight for localization loss.')

    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('-sched', '--scheduler', default='plateau', type=str, help='Learning rate scheduler')
    parser.add_argument('--patience', default=20, type=int, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--factor', default=0.5, type=float, help='Learning rate decay rate of ReduceLROnPlateau')

    parser.add_argument('--fine_tune', default=True, type=str2bool, help='Training feature extractor with start')
    parser.add_argument('--fine_tune_iter', default=0, type=int, help='Training feature extractor after this')
    parser.add_argument('--resume', default=True, type=str2bool, help='Resume training')
    parser.add_argument('--resume_epoch', default=0, type=int, help='Resume epoch for retraining')

    parser.add_argument('--loss_metric', default='weighted_sum', help='methods for multi-task learning',
                        choices=loss_metric_list)
    parser.add_argument('--focusing_factor', default=0.5, type=float, help='Focusing factor for multi-task loss')
    parser.add_argument('--potential', nargs='+', default=[0.6411, 0.8089, 0.8557], type=float,
                        help='Task potential of each task for achievement-based multi-task loss')
    parser.add_argument('--margin', default=0.15, type=float, help='margin for task potential')
    parser.add_argument('--loss_weight', nargs='+', default=None, type=float,
                        help='Initial loss weight [1.0, 1.0, 1.0]')
    parser.add_argument('--alpha', default=1.5, type=float, help='Restoring force')
    parser.add_argument('-max', '--max_epoch', default=200, type=int, help='Max epoch for training')
    parser.add_argument('-wi', '--warmup_epoch', default=2, type=int, help='Warmup epoch for training')
    parser.add_argument('-ci', '--cold_epoch', default=80, type=int, help='Decay epoch for learning rate')
    parser.add_argument('-fi', '--freeze_epoch', default=40, type=int, help='Freezing epoch for learning rate')

    parser.add_argument('--ckpt', default=None, help='Path of pretrained ckeckpoint')
    parser.add_argument('--teacher', default=None, nargs='+', help="Path of pretrained teachers' ckeckpoints")
    parser.add_argument('--output', default=None, help='Path of onnx output')
    parser.add_argument('--root_folder', required=True, help='Location to save checkpoint models')
    parser.add_argument('--subfolder', default='test')
    parser.add_argument('--save_freq', default=2, type=int)
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--display_freq', default=100, type=int)
    parser.add_argument('--remove_ckpt', default=True, type=str2bool, help='Set True to remove old check points')

    parser.add_argument('--mix_up', default=False, type=str2bool, help='Set true if uses mix up')
    parser.add_argument('--label_smoothing', default=0, type=float, help="Set zero if don't uses label smoothing")

    parser.add_argument('--dist_url', default='env://', type=str, help='Url used to set up distributed training')
    parser.add_argument('--world_size', type=int, default=-1, help='Total number of GPUs for training')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--amp', default=False, action='store_true', help='Set true if uses automatic mixed precision')

    arg_list = ['@' + arg if '.cfg' in arg else arg for arg in sys.argv[1:]]
    args = parser.parse_args(arg_list)

    assert not (args.cls_loss_type == 'focal' and args.cls_head == 'softmax'), \
        'Currently, we only support sigmoid head type for focal loss.'
    assert not (args.cls_loss_type == 'CE' and args.cls_head == 'sigmoid'), \
        'Currently, we only support softmax head type for CE loss.'

    if args.random_seed:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)  # if use multi-GPU
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

        # For deterministic
        # torch.backends.cudnn.deterministic = True

        # For performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    args.freeze_epoch = args.cold_epoch + args.freeze_epoch

    args.distributed = False
    args.size = (args.size[0], args.size[0]) if len(args.size) == 1 else tuple(args.size)
    assert len(args.size) == 2, 'Image should be two dimensional.'

    if len(args.training_size) == 0:
        args.training_size = args.size
    else:
        args.training_size = (args.training_size[0], args.training_size[0]) if len(args.training_size) == 1 \
            else tuple(args.training_size)

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) >= 1
        print(args.distributed)
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        print('world_size is %d, local_rank is %d' % (args.world_size, args.local_rank))
        if args.distributed and torch.cuda.device_count() > 1:
            print('Using more than one GPU per process in distributed mode is not allowed. Setting num_gpu to 1.')
            args.num_gpu = 1

    if args.distributed:
        import atexit
        args.num_gpu = 1
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method=args.dist_url, rank=args.local_rank,
                                world_size=args.world_size)
        atexit.register(dist.destroy_process_group)
        assert args.local_rank >= 0

    unit_batch_size = 32
    args.batch_size = args.batch_size if dist.is_initialized() else args.batch_size * torch.cuda.device_count()
    args.lr = (args.lr * args.batch_size / unit_batch_size)
    args.lr = args.lr * torch.cuda.device_count() if dist.is_initialized() else args.lr
    return args


def display_configs(args, net_to_eval, train_loader, test_loaders, root_folder, loss_function, logger=None):
    if not logger:
        logger = print

    if not torch.distributed.is_initialized() or not args.local_rank:
        logger(net_to_eval.__repr__())
        logger('\n===============================================')
        logger('- Architecture Configurations')
        logger('\t- Architecture: %s based %s' % (args.basenet, args.arch))
        logger('\t- Image size: {}'.format(args.size))
        logger('\t- Num_classes: {}'.format(net_to_eval.num_classes))
        logger('\t- GFlops: %.3f G' % (net_to_eval.flops / 1000000000))
        logger('\t- Total number of parameters: %.3f M' % (net_to_eval.params / 1000000))

        logger('\n- Training Configurations')
        logger('\t- Datasets')
        logger('\t\t- Training: ' + str(train_loader.dataset))
        logger('\t\t- Evaluation: ' + ', '.join(str(loader.dataset) for loader in test_loaders))
        logger('\t- Batch Size: {}'.format(args.batch_size))
        if args.sync_bn and torch.distributed.is_initialized():
            logger('\t\t- SyncBN is applied')

        logger('\t- Training Schedule')
        logger('\t\t- max epoch: %d' % args.max_epoch)
        logger('\t\t- fine_tune: {} fine_tune_iter: {}'.format(args.fine_tune, args.fine_tune_iter))

        logger('\t- Learning Rate')
        logger('\t\t- Initial learning rate: %.2e' % args.lr)
        logger('\t\t- learning scheduler: %s' % args.scheduler)
        if args.scheduler == 'multi-step':
            logger('\t\t\t- lr_warmup: {} cold_epoch: {} freeze_epoch: {}'.format(args.warmup_epoch, args.cold_epoch,
                                                                                  args.freeze_epoch))
        if args.scheduler == 'plateau':
            logger('\t\t\t- patience: {} factor: {}'.format(args.patience, args.factor))

        logger('\t- Optimizer Configs')
        logger('\t\t- Optimizer type: %s' % args.optim)
        logger('\t\t- weight decay: %s' % args.weight_decay)
        if args.tencent_trick:
            logger('\t\t- Tencent trick is applied')
        if args.multiple_lr > 1:
            logger('\t\t- x %s lr is applied to non-pretrained parameters' % args.multiple_lr)

        logger('\n- Loss Configurations')
        if len(args.app) > 1:
            if hasattr(loss_function, 'get_configs'):
                logger(loss_function.get_configs())
            else:
                logger('\t- Multi-Task Loss Metric: %s' % args.loss_metric)
                if args.loss_metric == 'weighted_sum':
                    logger('\t- loss_weight: {}'.format(args.loss_weight))
        if 'detection' in args.app:
            logger('\t- Detection Loss')
            logger('\t\t- classification loss type: %s' % (
                    ('IoU-aware ' if args.iou_aware_cls else '') + args.cls_loss_type))
            logger('\t\t- regression loss type: %s' % args.loc_loss_type)
            logger('\t\t- regression loss weight: %s' % args.loc_weight)
            logger('\t\t- mix_up: %s' % args.mix_up)
        logger('\t- label smoothing: %s' % args.label_smoothing)
        logger('\n- data will be stored in {}'.format(root_folder))

        if args.amp:
            logger('automatic mixed precision training is applied')
        logger('===============================================\n')
