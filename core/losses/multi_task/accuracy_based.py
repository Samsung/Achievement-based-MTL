import torch
import math as m
from abc import abstractmethod

from .multi_task import Losses
from .geometric import Geometric


def softmax(weights):
    return torch.softmax(torch.stack([value for value in weights.values()]), dim=-1)


def normalize(weights: dict):
    norm_const = 1 / sum(weights.values())
    return [value * norm_const for value in weights.values()]


class AccuracyBase:
    def __init__(self, applications, weight, *args, **kwargs):
        super(AccuracyBase, self).__init__(applications, weight, *args, **kwargs)
        self.kpi = {app: float('nan') for app in applications}
        self.momentum = 0.1

    def update_kpi(self, results):
        kpi = self._update_kpi(results)
        self._update_weight(kpi)

    @staticmethod
    def _update_kpi(results):
        key_dict = {
            'detection': ['mAP', 'mAP_2007', 'mAP_2012'],
            'segmentation': ['mIoU'],
            'depth': ['d1'],
            'normal': ['11.25'],
        }

        kpi = {key: (lambda x: sum(x) / len(x) if len(x) else None)([results[k] for k in value if k in results])
               for key, value in key_dict.items()}
        return kpi

    @abstractmethod
    def _update_weight(self, kpi: dict):
        pass


class DTPBase(AccuracyBase):
    """
    Dynamic Task Prioritization for Multi-Task Learning, ECCV 2018
    task_weight = - {(1 - current_accuracy) ** gamma} * log(current_accuracy)
    """
    def __init__(self, application, weight=None, focusing_factor=1):
        super(DTPBase, self).__init__(application, None)
        self.focusing = focusing_factor  # We recommend 0.5 for focusing factor of DTP
        self.weight = {app: torch.tensor([1], device='cuda' if torch.cuda.is_available() else 'cpu')
                       for i, app in enumerate(application)}
        self.name = 'DTP'

    def _update_weight(self, kpi):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for app, k in self.kpi.items():
            self.kpi[app] = kpi[app] if m.isnan(k) else self.momentum * kpi[app] + (1 - self.momentum) * self.kpi[app]
            focal_loss = -((1 - self.kpi[app]) ** self.focusing) * m.log(self.kpi[app] + 1e-20)
            self.weight[app] = torch.tensor(focal_loss, device=device)

    def get_configs(self):
        string = f'\t- Multi-Task Loss Metric: {self.name}\n'
        string += f'\t\t- focusing factor: {self.focusing}\n'
        string += f'\t\t- init weight: %s\n' % \
                  ', '.join(['%s - %.3f' % (key, value) for key, value in self.weight.items()])
        return string


class DTP(DTPBase, Losses):
    """
    Dynamic Task Prioritization for Multi-Task Learning, ECCV 2018
    task_weight = - {(1 - current_accuracy) ** gamma} * log(current_accuracy)
    """


class DTPG(DTPBase, Geometric):
    """
    DTP with a Weighted Geometric Loss
    """
    def __init__(self, application, weight=None, focusing_factor=1):
        super(DTPG, self).__init__(application, weight, focusing_factor)
        self.name = 'DTP-G'


class MTLwFW(DTP):
    """
    Multi-Task Loss with Focal Weights
    task_weight = (1 - current_accuracy) ** focusing_factor
    """
    def __init__(self, application, weight=None, focusing_factor=2, normal_type='normal'):
        super(MTLwFW, self).__init__(application, focusing_factor)
        self._weight_norm = softmax if normal_type == 'softmax' else normalize
        normalized_weight = self._weight_norm(self.weight)
        self.weight = {key: value for key, value in zip(self.weight.keys(), normalized_weight)}

    def get_configs(self):
        string = '\t- Multi-Task Loss Metric: FMTL\n'
        string += '\t\t- focusing factor: %.2f\n' % self.focusing
        string += '\t\t- init weight: %s\n' % \
                  ', '.join(['%s - %.3f' % (key, value) for key, value in self.weight.items()])
        return string

    def _update_weight(self, kpi):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for app, k in self.kpi.items():
            self.kpi[app] = kpi[app] if m.isnan(k) else self.momentum * kpi[app] + (1 - self.momentum) * k
            self.weight[app] = torch.tensor(((1 - self.kpi[app]) ** self.focusing), device=device)
        normalized_weight = self._weight_norm(self.weight)
        self.weight = {key: value for key, value in zip(self.weight.keys(), normalized_weight)}


class AMTL(AccuracyBase):
    """
    Achievement-based Multi-Task Loss
    task_weight = {(task_potential - current_accuracy) / task_potential} ** focusing_factor
                = (1 - current_accuracy / task_potential) ** focusing_factor
    """
    def __init__(self, application, weight=None, potential=None, focusing_factor=2, margin=0.05, normal_type='softmax'):
        assert len(application) == len(potential), f'{self.__class__.__name__} requires a task potential for each task'
        self.margin = margin
        self.focusing = focusing_factor
        self.potential = {app: potential[i] * (1 + margin) if potential else 1.0
                          for i, app in enumerate(application)}
        self._weight_norm = softmax if normal_type == 'softmax' else normalize

        # Task weight initialization considering potential
        # weight = [(1 + margin - value) for key, value in self.potential.items()]
        super(AMTL, self).__init__(application, weight)

        normalized_weight = self._weight_norm(self.weight)
        self.weight = {key: value for key, value in zip(self.weight.keys(), normalized_weight)}

    def get_configs(self):
        string = '\t\t- task potential: %s\n' % \
                  (', '.join(['%s - %.3f' % (key, value) for key, value in self.potential.items()]))
        string += '\t\t- focusing factor: %.2f\n' % self.focusing
        string += '\t\t- margin: %.2f\n' % self.margin
        string += '\t\t- init weight: %s\n' % \
                  ', '.join(['%s - %.3f' % (key, value) for key, value in self.weight.items()])
        return string

    def _update_weight(self, kpi):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for app, k in self.kpi.items():
            self.kpi[app] = kpi[app] if m.isnan(k) else self.momentum * kpi[app] + (1 - self.momentum) * k
            self.weight[app] = torch.tensor((1 - self.kpi[app] / self.potential[app]) ** self.focusing, device=device)
        normalized_weight = self._weight_norm(self.weight)
        self.weight = {key: value for key, value in zip(self.weight.keys(), normalized_weight)}


class AMTLG(AMTL, Geometric):
    """
    Achievement-based multi-task loss with Geometric mean
    """

    def get_configs(self):
        string = super(AMTLG, self).get_configs()
        string = '\t- Multi-Task Loss Metric: AMTL with Geometric Mean\n' + string
        return string


class AMTLA(AMTL, Losses):
    """
    Achievement-based multi-task loss with Arithmetic mean
    """

    def get_configs(self):
        string = super(AMTLA, self).get_configs()
        string = '\t- Multi-Task Loss Metric: AMTL with Arithmetic Mean\n' + string
        return string

