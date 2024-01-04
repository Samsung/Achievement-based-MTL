import torch
import torch.nn.functional as F
from ..loss import Loss


class DepthEstimationLoss(Loss):
    def __init__(self, loss_type, variance_focus=0.85):
        super(DepthEstimationLoss, self).__init__()

        supported_losses = ['l1_loss', 'CE']
        assert loss_type in supported_losses, '%s loss is not supported for depth estimation' % loss_type
        if loss_type == 'l1_loss':
            self.variance_focus = variance_focus
            self.loss = self._linear_depth
        elif loss_type == 'CE':
            self.loss = self._cls_depth
        self.loss_dict['depth'] = 0
        self.min_depth, self.max_depth = 0.001, 10

    def loss_with_gt(self, preds, gt):
        loss = 0
        if 'depth' in gt:
            mask = torch.logical_and((gt['depth'] > self.min_depth), (gt['depth'] < self.max_depth))
            loss += self.loss(preds, gt, mask)
            self.loss_dict['depth'] += loss.item()
        return loss

    def _linear_depth(self, preds, gt, mask):
        return F.l1_loss(preds['depth'][mask], gt['depth'][mask])

    def _cls_depth(self, preds, gt, mask):
        cls, residual = self.depth_to_label(gt['depth'][mask], preds['depth'].shape[1])

        mask = mask.squeeze(dim=1)
        cls_pred = preds['depth'].permute(0, 2, 3, 1)[mask]
        cls_loss = F.cross_entropy(cls_pred, cls, reduction='mean')

        cls = cls.unsqueeze(dim=-1)
        reg_pred = torch.gather(preds['depth_residual'].permute(0, 2, 3, 1)[mask], 1, cls)
        residual = torch.gather(residual, 1, cls)
        reg_loss = F.smooth_l1_loss(reg_pred, residual, reduction='mean')
        return cls_loss + reg_loss

    def loss_with_pseudo(self, preds, pseudo):
        loss = 0
        if 'depth' in pseudo:
            loss += F.smooth_l1_loss(preds['depth'], pseudo['depth'])
            self.loss_dict['depth'] += loss.item()
        return loss

    def depth_to_label(self, depth, bins):
        with torch.no_grad():
            log_interval = torch.stack([torch.log(torch.tensor(self.min_depth)) +
                                        torch.log(torch.tensor(self.max_depth / self.min_depth)) / bins * i
                                        for i in range(bins + 1)])
            interval = torch.exp(log_interval)
            mid = [(log_interval[i] + log_interval[i + 1]) / 2 for i in range(bins)]

            cls = torch.bucketize(depth, torch.Tensor(interval)) - 1
            residual = torch.stack([(torch.log(depth) - mid[i]) /
                                    (log_interval[i + 1] - log_interval[i])
                                    for i in range(bins)], dim=-1)
        return cls, residual
