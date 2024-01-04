import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = 'sum'):
        super().__init__()
        self.reduction = reduction

    def __call__(self, conf, target, iou=None, smoothing=True):
        lsm = F.log_softmax(conf, 1)
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

        if iou is None:
            loss = -likelihood
        else:
            if smoothing:
                smooth = (1 - iou) / (conf.shape[-1] - 1)
                loss = -((iou - smooth) * likelihood + smooth * lsm.sum(-1))
            else:
                loss = -iou * likelihood

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
