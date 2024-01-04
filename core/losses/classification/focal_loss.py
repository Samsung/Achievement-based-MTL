import torch
import torch.nn as nn
import torch.nn.functional as F


def iou_aware_sigmoid_focal_loss(inputs, targets, alpha, gamma, thr=0, reduction="none"):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = (torch.abs(targets - p) ** gamma) * ce_loss
    if alpha >= 0:
        alpha_t = torch.where(targets <= thr, 1 - alpha, alpha)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class FocalLoss(nn.BCELoss):
    """ sigmoid Focal loss for binary classification tasks on imbalanced datasets """

    def __init__(self, alpha=0.25, gamma=2, reduction='none'):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets, iou=None, smoothing=True):
        encoded_target = torch.zeros_like(inputs)
        encoded_target[(targets > 0), targets[targets > 0] - 1] = 1 if iou is None else iou.to(inputs.dtype)
        return iou_aware_sigmoid_focal_loss(inputs, encoded_target, self.alpha, self.gamma, reduction=self.reduction)
