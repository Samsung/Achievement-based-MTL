import torch

optimizer_list = ['sgd', 'rms_prop', 'adam', 'adamw']


def build_optimizer(args, net_without_ddp):
    params = get_model_params(net_without_ddp, args.lr, args.tencent_trick, args.multiple_lr)
    return build_optimizer_from_params(args, params)


def build_optimizer_from_params(args, params):
    if args.optim == 'sgd':
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'rms_prop':
        return torch.optim.RMSprop(params, lr=args.lr, alpha=0.9, momentum=args.momentum,
                                   weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        assert 'No matched optimizer with %s' % args.optim


def get_model_params(model, lr, tencent_trick=False, multiple=10):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    if multiple > 1:
        if tencent_trick:
            pretrained_decay, pretrained_no_decay = [], []
            for name, param in model.get_pretrained_params():
                if len(param.shape) == 1 or name.endswith(".bias"):
                    pretrained_no_decay.append(param)
                else:
                    pretrained_decay.append(param)

            scratch_decay, scratch_no_decay = [], []
            for name, param in model.get_from_scratch_params():
                if len(param.shape) == 1 or name.endswith(".bias"):
                    scratch_no_decay.append(param)
                else:
                    scratch_decay.append(param)
            return [{'params': pretrained_decay},
                    {'params': pretrained_no_decay, 'weight_decay': 0.0},
                    {'params': scratch_decay, 'lr': multiple * lr},
                    {'params': scratch_no_decay, 'lr': multiple * lr, 'weight_decay': 0.0}]
        else:
            pretrained = [param for name, param in model.get_pretrained_params()]
            from_scratch = [param for name, param in model.get_from_scratch_params()]
            return [{'params': pretrained},
                    {'params': from_scratch, 'lr': multiple * lr}]
    else:
        if tencent_trick:
            decay, no_decay = [], []
            for name, param in model.named_parameters():
                if len(param.shape) == 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [{'params': no_decay, 'weight_decay': 0.0},
                    {'params': decay}]
        else:
            return model.parameters()
