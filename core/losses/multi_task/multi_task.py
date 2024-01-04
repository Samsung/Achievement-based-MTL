import torch
from ..task_losses import DetectionLoss, SegmentationLoss, DepthEstimationLoss, SurfaceNormalLoss


class Losses:
    def __init__(self, application, weight=None):
        self.losses = dict()
        self.cur_loss = dict()

        if weight is None:
            weight = [1 / len(application) for _ in range(len(application))]
        assert len(application) == len(weight), 'num of loss weights should be same with the num of applications'
        self.weight = {app: torch.tensor(weight[i], device='cuda' if torch.cuda.is_available() else 'cpu')
                       for i, app in enumerate(application)}

    def register_loss(self, application, loss_type, **kwargs):
        task_loss_dict = {
            'detection': DetectionLoss,
            'segmentation': SegmentationLoss,
            'depth': DepthEstimationLoss,
            'normal': SurfaceNormalLoss
        }
        assert application in task_loss_dict, '%s is not supported yet' % application
        self.losses[application] = task_loss_dict[application](loss_type, **kwargs)

    def __call__(self, preds, gt=None, pseudo=None):
        if gt is None:
            gt = dict()
        if pseudo is None:
            pseudo = dict()

        loss = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
        for key, function in self.losses.items():
            self.cur_loss[key] = function(preds, gt, pseudo) if key in gt or key in pseudo \
                else sum([pred.sum() for pred in preds[key]]) * 0
            loss += self.cur_loss[key] * self.weight[key].item()
        return loss

    def get_cur_loss(self):
        loss = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
        for key, function in self.losses.items():
            loss += self.cur_loss[key] * self.weight[key].item()
        return loss

    def __str__(self):
        report, loss_dict, total_loss = '', dict(), 0
        for app, loss in self.losses.items():
            loss_dict[app] = 0
            for name, value in loss.items():
                loss_dict[app] += value
            report += '%s: %3.3f ' % (app, loss_dict[app])
            total_loss += loss_dict[app] * self.weight[app]
        return report + 'total: %3.3f' % total_loss

    def items(self):
        total_loss, losses = 0, dict()
        for app, loss in self.losses.items():
            for key, value in loss.items():
                losses[key] = value
                total_loss += value * self.weight[app].item()
        losses['total_loss'] = total_loss
        return losses

    def get_weights(self):
        weights = dict()
        for app, weight in self.weight.items():
            weights[app+'_weight'] = weight.item()
        return weights

    def clear(self):
        for app, loss in self.losses.items():
            loss.clear()


class WeightedSum(Losses):
    def __init__(self, application, weight=None):
        super(WeightedSum, self).__init__(application, weight)
