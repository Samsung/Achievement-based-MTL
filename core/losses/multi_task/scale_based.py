import torch
from .multi_task import Losses


class DWA(Losses):
    """
    Dynamic Weight Averaging is proposed in End-to-End Multi-Task Learning with Attention (CVPR 2019)
    This implementation is based on the official github repository
    https://github.com/lorenmt/mtan/blob/db5681a6b3451e35b297cc714b1abda5b1540fd6/im2im_pred/utils.py#L139-L147
    """

    def __init__(self, application, weight=None, temperature=2):
        weight = [1 for _ in range(len(application))]
        super(DWA, self).__init__(application, weight)
        self._temperature = temperature
        self._prev_losses = dict()

    def clear(self):
        self._update_weights()  # update weights for every display frequency (or epoch)
        super(DWA, self).clear()

    def _update_weights(self):
        if not self._prev_losses:
            self._set_prev_losses()
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        weight = {app: torch.tensor(sum(loss.values()) / self._prev_losses[app], device=device).exp()
                  for app, loss in self.losses.items()}
        norm_const = len(weight) / sum(weight.values())
        self.weight = {app: w * norm_const for app, w in weight.items()}
        self._set_prev_losses()

    def _set_prev_losses(self):
        for app, loss in self.losses.items():
            if loss.count:
                self._prev_losses[app] = sum(loss.values()) / self._temperature


class RLW(Losses):
    """
    Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (TMLR 2022)
    """

    def __init__(self, application, weight=None):
        weight = [1 for _ in range(len(application))]
        super(RLW, self).__init__(application, weight)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __call__(self, preds, gt=None, pseudo=None):
        self._rand_task_weights()
        return super(RLW, self).__call__(preds, gt, pseudo)

    def _rand_task_weights(self):
        weight = torch.softmax(torch.randn(len(self.weight)), dim=-1).to(self.device)
        for i, key in enumerate(self.weight.keys()):
            self.weight[key] = weight[i]
