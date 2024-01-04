import torch
import torch.nn as nn
import torch.nn.init as init


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale, trainable=True):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels), requires_grad=trainable)
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        # x /= norm
        x = torch.div(x, norm)
        expand = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = expand * x
        return out
