from abc import *

import torch
import torch.distributed as dist


class Loss(metaclass=ABCMeta):
    def __init__(self):
        self.count = 0
        self.loss_dict = {}

    def __call__(self, preds, gt=None, pseudo=None):
        self.count += 1
        loss = self.loss_with_gt(preds, gt) + self.loss_with_pseudo(preds, pseudo)
        return loss

    @abstractmethod
    def loss_with_gt(self, preds, gt):
        pass

    @abstractmethod
    def loss_with_pseudo(self, preds, pseudo):
        pass

    def items(self):
        if self.count == 0:
            return {key: 0 for key, value in self.loss_dict.items()}.items()

        if dist.is_initialized():
            losses = torch.Tensor([value for _, value in self.loss_dict.items()]).cuda()
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses = losses / dist.get_world_size() / self.count
            return {key: losses[i].item() for i, (key, value) in enumerate(self.loss_dict.items())}.items()
        else:
            return {key: value / self.count for key, value in self.loss_dict.items()}.items()

    def keys(self):
        return [key for key, _ in self.loss_dict.items()]

    def values(self):
        return self.loss_dict.values()

    def __getitem__(self, item):
        return self.loss_dict[item].item()

    def clear(self):
        self.count = 0
        self.loss_dict = {key: 0 for key, value in self.loss_dict.items()}
