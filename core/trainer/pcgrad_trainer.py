import torch
import random
import numpy as np
from scipy.optimize import minimize

from .trainer import Trainer


class PCGradTrainer(Trainer):
    """
    This code reproduced PCGrad which is proposed in 'Gradient Surgery for Multi-Task Learning' at NeurIPS 2020.
    The PCGrad implementation was based on https://github.com/WeiChengTseng/Pytorch-PCGrad.
    The official TensorFlow implementation: https://github.com/tianheyu927/PCGrad.
    """

    def __init__(self, args, net, data_loader, teachers=None):
        super(PCGradTrainer, self).__init__(args, net, data_loader, teachers)

    def _backward(self, loss, scaler, epoch=None):
        self.optimizer.zero_grad(set_to_none=True)
        losses = [weight * cur_loss for weight, cur_loss
                  in zip(self.loss_function.weight.values(), self.loss_function.cur_loss.values())]
        pc_backward(losses, self.optimizer, scaler)
        scaler.step(self.optimizer)
        scaler.update()
        self.scheduler.step(epoch)


class CAGradTrainer(Trainer):
    """
    This code reproduced to CAGrad proposed in 'Conflict-Averse Gradient Descent for Multitask Learning' at NeurIPS 2021
    The official CAGrad implementation: https://github.com/Cranial-XIX/CAGrad
    """

    def __init__(self, args, net, data_loader, teachers=None):
        super(CAGradTrainer, self).__init__(args, net, data_loader, teachers)

    def _backward(self, loss, scaler, epoch=None):
        self.optimizer.zero_grad(set_to_none=True)
        losses = [weight * cur_loss for weight, cur_loss
                  in zip(self.loss_function.weight.values(), self.loss_function.cur_loss.values())]
        ca_backward(losses, self.optimizer, scaler)
        scaler.step(self.optimizer)
        scaler.update()
        self.scheduler.step(epoch)


def pc_backward(losses, optimizer, scaler):
    """
    calculate the gradient of the parameters

    input:
    - objectives: a list of objectives
    """

    grads, shapes, has_grads = pack_grad(losses, optimizer, scaler)
    grads = project_conflicting(grads, has_grads)
    grads = unflatten_grad(grads, shapes[0])
    set_grad(grads, optimizer)
    return


def ca_backward(losses, optimizer, scaler):
    """
    calculate the gradient of the parameters

    input:
    - objectives: a list of objectives
    """

    grads, shapes, has_grads = pack_grad(losses, optimizer, scaler)
    grads = cagrad(grads)
    grads = unflatten_grad(grads, shapes[0])
    set_grad(grads, optimizer)
    return


def cagrad(grads, alpha=0.5, rescale=1):
    num_tasks = len(grads)
    grads = torch.stack(grads, dim=-1)
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean()+1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(num_tasks) / num_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha*g0_norm+1e-8).item()

    def objfn(x):
        return (x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) +
                c * np.sqrt(x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha**2)
    else:
        return g / (1 + alpha)


def pack_grad(losses, optimizer, scaler):
    """
    pack the gradient of the parameters of the network for each objective

    output:
    - grad: a list of the gradient of the parameters
    - shape: a list of the shape of the parameters
    - has_grad: a list of mask represent whether the parameter has gradient
    """

    grads, shapes, has_grads = [], [], []
    num_task = len(losses)
    for i, loss in enumerate(losses, 1):
        scaler.scale(loss).backward(retain_graph=(i != num_task))
        grad, shape, has_grad = retrieve_grad(optimizer)
        grads.append(flatten_grad(grad))
        has_grads.append(flatten_grad(has_grad))
        shapes.append(shape)
    return grads, shapes, has_grads


def retrieve_grad(optimizer):
    """
    get the gradient of the parameters of the network with specific
    objective

    output:
    - grad: a list of the gradient of the parameters
    - shape: a list of the shape of the parameters
    - has_grad: a list of mask represent whether the parameter has gradient
    """

    grad, shape, has_grad = [], [], []
    for group in optimizer.param_groups:
        for p in group['params']:
            # if p.grad is None: continue
            # tackle the multi-head scenario
            if p.grad is None:
                shape.append(p.shape)
                grad.append(torch.zeros_like(p).to(p.device))
                has_grad.append(torch.zeros_like(p).to(p.device))
                continue
            shape.append(p.grad.shape)
            grad.append(p.grad.detach())
            has_grad.append(torch.ones_like(p).to(p.device))
    return grad, shape, has_grad


def flatten_grad(grads):
    _flatten_grad = torch.cat([g.flatten() for g in grads])
    return _flatten_grad


def project_conflicting(grads, has_grads, reduction='mean'):
    shared = torch.stack(has_grads).prod(0).bool()
    pc_grad = [grad.detach() for grad in grads]
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= g_i_g_j * g_j / (g_j.norm() ** 2)
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    if reduction == 'mean':
        merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
    elif reduction == 'sum':
        merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
    else:
        exit('invalid reduction method')

    merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
    return merged_grad


def unflatten_grad(grads, shapes):
    _unflatten_grad, idx = [], 0
    for shape in shapes:
        length = shape.numel()
        _unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
        idx += length
    return _unflatten_grad


def set_grad(grads, optimizer):
    """
    set the modified gradients to the network
    """

    idx = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            # if p.grad is None: continue
            p.grad = grads[idx]
            idx += 1
    return
