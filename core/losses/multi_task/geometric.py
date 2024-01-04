import torch
from .multi_task import Losses


class Geometric(Losses):
    """
    MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning, CVPRW 2019
    """
    def __init__(self, application, weight=None):
        super(Geometric, self).__init__(application, weight)

    def __call__(self, preds, gt=None, pseudo=None):
        if gt is None:
            gt = dict()
        if pseudo is None:
            pseudo = dict()

        loss = torch.ones(1, device='cuda' if torch.cuda.is_available() else 'cpu')
        for key, function in self.losses.items():
            self.cur_loss[key] = function(preds, gt, pseudo) if key in gt or key in pseudo \
                else sum([pred.sum() for pred in preds[key]]) * 0
            loss *= torch.pow(self.cur_loss[key], self.weight[key])
        return loss

    def get_cur_loss(self):
        loss = torch.ones(1, device='cuda' if torch.cuda.is_available() else 'cpu')
        for key, function in self.losses.items():
            loss *= torch.pow(self.cur_loss[key], self.weight[key])
        return loss

    def __str__(self):
        report, loss_dict, total_loss = '', dict(), 1
        for app, loss in self.losses.items():
            loss_dict[app] = 0
            for name, value in loss.items():
                loss_dict[app] += value
            report += '%s: %3.3f ' % (app, loss_dict[app])
            total_loss *= loss_dict[app]
        return report + 'total: %3.3f' % total_loss ** (1 / len(self.losses))

    def items(self):
        total_loss, losses = 1, dict()
        for app, loss in self.losses.items():
            task_loss = 0
            for key, value in loss.items():
                losses[key] = value
                task_loss += value
            total_loss *= task_loss ** self.weight[app].item()
        losses['total_loss'] = total_loss
        return losses


class DWAG(Geometric):
    """
    Dynamic Weight Averaging with a Weighted Geometric Mean
    """

    def __init__(self, application, weight=None, temperature=2):
        weight = [1 for _ in range(len(application))]
        super(DWAG, self).__init__(application, weight)
        self._temperature = temperature
        self._prev_losses = dict()

    def clear(self):
        self._update_weights()  # update weights for every display frequency (or epoch)
        super(DWAG, self).clear()

    def _update_weights(self):
        if not self._prev_losses:
            self._set_prev_losses()
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        weight = {app: torch.tensor(sum(loss.values()) / self._prev_losses[app], device=device).exp()
                  for app, loss in self.losses.items()}
        norm_const = 1 / sum(weight.values())
        self.weight = {app: w * norm_const for app, w in weight.items()}
        self._set_prev_losses()

    def _set_prev_losses(self):
        for app, loss in self.losses.items():
            if loss.count:
                self._prev_losses[app] = sum(loss.values()) / self._temperature


class RLWG(Geometric):
    """
    Random Loss Weighting with a Weighted Geometric Mean
    """

    def __init__(self, application, weight=None):
        weight = [1 for _ in range(len(application))]
        super(RLWG, self).__init__(application, weight)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __call__(self, preds, gt=None, pseudo=None):
        self._rand_task_weights()
        return super(RLWG, self).__call__(preds, gt, pseudo)

    def _rand_task_weights(self):
        weight = torch.softmax(torch.randn(len(self.weight)), dim=-1).to(self.device)
        for i, key in enumerate(self.weight.keys()):
            self.weight[key] = weight[i]
