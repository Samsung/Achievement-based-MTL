from ..loss import Loss

import torch
import torch.nn.functional as F


class SurfaceNormalLoss(Loss):
    def __init__(self, loss_type):
        super(SurfaceNormalLoss, self).__init__()
        self.loss_dict['normal'] = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def loss(pred, gt):
        mask = gt.abs().sum(dim=1) != 0
        loss = 1 - (pred * gt).sum(dim=1)[mask].mean()
        return loss

    def loss_with_gt(self, preds, gt):
        loss = 0
        if 'normal' in gt:
            loss = self.loss(preds['normal'], gt['normal'])
            if loss.isnan():
                loss = preds['normal'].sum() * 0
            self.loss_dict['normal'] += loss.item()
        return loss

    def loss_with_pseudo(self, preds, pseudo):
        loss = 0
        if 'normal' in pseudo:
            target = F.softmax(pseudo['normal'], dim=1)
            loss += self.loss(preds['normal'], target)
            self.loss_dict['normal'] += loss.item()
        return loss
