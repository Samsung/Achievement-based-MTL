import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineDecay(_LRScheduler):
    def __init__(self, optimizer, T_0, scale=0.5, T_mult=0.5, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.scale = scale
        self.quotient = 0
        super(CosineDecay, self).__init__(optimizer, last_epoch)
        self.T_cur = last_epoch

    def get_lr(self):
        return [self.eta_min + (base_lr*self.scale**self.quotient - self.eta_min)
                * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every update, i.e. if one epoch has 10 iterations
        (number_of_train_examples / batch_size), we should call SGDR.step(0.1), SGDR.step(0.2), etc.

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = SGDR(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.quotient += 1
            if self.last_epoch == self.T_0:
                self.T_i = self.T_i * self.T_mult
                self.eta_min *= self.scale
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class PolyDecay(_LRScheduler):
    def __init__(self, optimizer, power, max_epoch, last_epoch=-1):
        self.base_lr = optimizer.param_groups[0]['lr']
        self.power = power
        self.max_epoch = max_epoch
        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.base_lr * (1 - float(self.last_epoch) / self.max_epoch) ** self.power]
