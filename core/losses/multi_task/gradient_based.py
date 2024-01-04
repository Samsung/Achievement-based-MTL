import torch
from abc import abstractmethod
from .multi_task import Losses
from .min_norm_solvers import MinNormSolver
import math as m


def normalize(weights: dict):
    norm_const = 1 / sum(weights.values())
    return [value * norm_const for value in weights.values()]


class GradientBase(Losses):
    def __init__(self, application, weight=None):
        super().__init__(application=application, weight=weight)

    def __call__(self, preds, gt=None, pseudo=None):
        loss = super().__call__(preds, gt, pseudo)

        if loss.requires_grad:  # only for training
            self._update_weights()
            loss = 0
            for key, cur_loss in self.cur_loss.items():
                loss += self.weight[key] * cur_loss
        return loss

    @abstractmethod
    def _update_weights(self):
        pass


class GradNorm(GradientBase):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks, ICML 2018
    """
    def __init__(self, application, weight=None, shared_weights=None, lr=0.01):
        assert shared_weights is not None, 'GradNorm requires last shared weights'
        super().__init__(application=application, weight=weight)

        self._alpha = 1.5
        self._init_loss = None
        self.shared_weights = shared_weights

        for weight in self.weight.values():
            weight.requires_grad = True
        params = [{'params': self.weight[app]} for app in application]
        self.grad_optimizer = torch.optim.SGD(params, lr=lr, momentum=0.1)

    def _update_weights(self):
        if self._init_loss is None:
            self._init_loss = {key: value for key, value in self.cur_loss.items()}

        task_num = len(self.losses)
        grad_norms, inverse_train_rate = dict(), dict()
        for key, function in self.losses.items():
            grad = torch.autograd.grad(self.cur_loss[key], self.shared_weights, retain_graph=True)
            grad_norms[key] = torch.norm(torch.mul(self.weight[key], grad[0]))
            inverse_train_rate[key] = self.cur_loss[key].detach() / self._init_loss[key]
        mean_norm = (sum(grad_norms.values()) / task_num)

        denominator = sum(inverse_train_rate.values())
        denominator = denominator / task_num + torch.finfo(denominator.dtype).eps
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        grad_norm_loss = torch.zeros(1, device=device)
        for key in inverse_train_rate.keys():
            inverse_train_rate[key] /= denominator
            constant_term = (mean_norm * (inverse_train_rate[key] ** self._alpha)).detach()
            grad_norm_loss += torch.abs(grad_norms[key] - constant_term)

        self.grad_optimizer.zero_grad()
        grad_norm_loss.backward()
        self.grad_optimizer.step()

        norm_const = task_num / sum(self.weight.values())
        for key, val in self.weight.items():
            self.weight[key].data *= norm_const


class MGDA(GradientBase):
    """
    Multiple-Gradient Descent Algorithm proposed in Multi-Task Learning as Multi-Objective Optimization at NeurIPS 2018
    The core of MGDA is implemented in .min_norm_solver.py which is taken from official repository
    """
    def __init__(self, application, weight=None, shared_weights=None, use_fw=True):
        assert shared_weights is not None, 'MGDA requires shared weights'
        super().__init__(application=application, weight=None)
        self.shared_weights = [param for param in shared_weights]
        self.find_min_norm_element = MinNormSolver.find_min_norm_element_FW if use_fw \
            else MinNormSolver.find_min_norm_element

    def _update_weights(self):
        grads = dict()
        for app, task_loss in self.cur_loss.items():
            task_loss.backward(retain_graph=True)
            grads[app] = []
            for name, param in self.shared_weights:
                if param.grad is not None:
                    grads[app].append(param.grad.detach())
                    param.grad = None

        sol = self.find_min_norm_element(list(grads.values()))
        self.weight = {app: weight for app, weight in zip(self.cur_loss.keys(), sol)}
        # if torch.tensor(sol).sum().isnan():
        #     self.weight = {app: torch.zeros(1).cuda() for app in self.cur_loss.keys()}
        # else:
        #     self.weight = {app: weight for app, weight in zip(self.cur_loss.keys(), sol)}


class IMTLG(GradientBase):
    """
    Towards Impartial Multi-Task Learning, ICLR 2021
    """
    def __init__(self, application, weight=None, shared_weights=None):
        assert shared_weights is not None, 'IMTLG requires last shared weights'
        super().__init__(application=application, weight=weight)
        self.shared_weights = shared_weights
        self.init_loss = dict()

    def _update_weights(self):
        grad, unit = self._get_grad_and_unit()
        task_num = len(self.losses)
        grad_diff = torch.stack([grad[0] - grad[i + 1] for i in range(task_num - 1)], dim=0)  # [Task_N, Len]
        unit_diff = torch.stack([unit[0] - unit[i + 1] for i in range(task_num - 1)], dim=0)
        unit_diff_t = unit_diff.transpose(1, 0)

        dtype = unit_diff_t.dtype
        alpha = torch.matmul(torch.matmul(grad[0], unit_diff_t),
                             torch.inverse(torch.matmul(grad_diff, unit_diff_t).float()).to(dtype))
        alpha = torch.cat([(1 - alpha.sum()).unsqueeze(0), alpha])
        self.weight = {key: alpha[i] for i, key in enumerate(self.losses.keys())}

    @abstractmethod
    def _get_grad_and_unit(self):
        grad, unit = [], []
        for key, loss in self.cur_loss.items():
            _grad = torch.autograd.grad(loss, self.shared_weights, retain_graph=True)[0].view(-1)
            grad.append(_grad)
            unit.append(_grad / (torch.norm(_grad) + torch.finfo(_grad.dtype).eps))
        return grad, unit


class IMTL(IMTLG):
    """
    Towards Impartial Multi-Task Learning, ICLR 2021
    """
    def __init__(self, application, weight=None, shared_weights=None):
        assert shared_weights is not None, 'IMTL requires last shared weights'
        super().__init__(application=application, weight=weight, shared_weights=shared_weights)

        self.scaled_loss = dict()
        self.loss_scale = {key: torch.nn.Parameter(value.zero_(), requires_grad=True)
                           for key, value in self.weight.items()}

    def __call__(self, preds, gt=None, pseudo=None):
        loss = super(IMTL, self).__call__(preds, gt, pseudo)

        if loss.requires_grad:
            shared_loss, task_loss = self._get_losses()
            return shared_loss, task_loss
        return loss

    @abstractmethod
    def _get_grad_and_unit(self):
        grad, unit = [], []
        for key, loss in self.cur_loss.items():
            self.scaled_loss[key] = self.loss_scale[key].clamp(max=88).exp() * loss - self.loss_scale[key]
            _grad = torch.autograd.grad(self.scaled_loss[key], self.shared_weights, retain_graph=True)[0].view(-1)
            grad.append(_grad)
            unit.append(_grad / (torch.norm(_grad) + torch.finfo(_grad.dtype).eps))
        return grad, unit

    def _get_losses(self):
        self._update_weights()

        task_loss = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
        for key, loss in self.scaled_loss.items():
            task_loss += (1 - self.weight[key]) * loss

        shared_loss = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
        for key, loss in self.scaled_loss.items():
            shared_loss += self.weight[key] * loss

        return shared_loss, task_loss


class NashMTL(GradientBase):
    def __init__(self, application, weight=None, shared_weights=None):
        assert shared_weights is not None, 'NashMTL requires last shared weights'
        super().__init__(application=application, weight=weight)
        self.shared_weights = shared_weights
        self.init_loss = dict()

    def _update_weights(self):
        pass

    @abstractmethod
    def _get_grad_vector(self):
        return [torch.autograd.grad(loss, self.shared_weights, retain_graph=True)[0].view(-1)
                for key, loss in self.cur_loss.items()]
