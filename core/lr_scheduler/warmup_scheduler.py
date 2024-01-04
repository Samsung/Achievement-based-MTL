from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, start_epoch, warmup_epoch, last_epoch=-1, after_scheduler=None):
        self.end_epoch = start_epoch + warmup_epoch
        self.start_epoch = start_epoch
        self.total_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        if hasattr(after_scheduler, 'last_epoch'):
            self.after_scheduler.last_epoch = last_epoch
        self.after_scheduler.last_epoch = warmup_epoch
        self.finished = False

        if start_epoch:
            last_epoch = start_epoch
        super(GradualWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.end_epoch:
            self.finished = True
            return [base_lr for base_lr in self.base_lrs]
        return [base_lr * (self.last_epoch - self.start_epoch) / self.total_epoch for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch = epoch if epoch != 0 else 1
        if metrics is not None:
            self.after_scheduler.step(metrics, epoch)

    def step(self, epoch=None, metrics=None):
        if not self.finished:
            return super(GradualWarmupScheduler, self).step(epoch)
        else:
            if type(self.after_scheduler) != ReduceLROnPlateau:
                # the 'epoch' argument of 'step' member function will be deprecated.
                # However, we requires that argument so that step should be multiplied by the number of GPUs
                # for training with multi-GPUs. So we directly manipulates last_epoch of after scheduler.
                self.last_epoch = epoch if epoch else self.last_epoch + 1
                self.after_scheduler.step(epoch)
            else:
                self.step_ReduceLROnPlateau(metrics, epoch)
