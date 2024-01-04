from ..loss import Loss

import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(Loss):
    def __init__(self, loss_type, smoothing=0, ignore_index=255):
        super(SegmentationLoss, self).__init__()

        supported_losses = ['cross_entropy']
        assert loss_type in supported_losses, '%s is not supported for segmentation' % loss_type
        if loss_type == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=smoothing)
        self.loss_dict['segmentation'] = 0

    def loss_with_gt(self, preds, gt):
        loss = 0
        if 'segmentation' in gt:
            loss = self.loss(preds['segmentation'], gt['segmentation'])
            if loss.isnan():
                loss = preds['segmentation'].sum() * 0
            self.loss_dict['segmentation'] += loss.item()
        return loss

    def loss_with_pseudo(self, preds, pseudo):
        loss = 0
        if 'segmentation' in pseudo:
            target = F.softmax(pseudo['segmentation'], dim=1)
            loss += F.cross_entropy(preds['segmentation'], target, reduction='mean')
            self.loss_dict['segmentation'] += loss.item()
        return loss
