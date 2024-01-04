import torch
from copy import deepcopy
from thop import profile
from thop.vision.basic_hooks import zero_ops
from nn.modules.conv2d import Conv2d, Conv2dDynamicSamePadding, Conv2dStaticSamePadding


def count_gops_convNd(m, x, y: torch.Tensor):
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops * 2 + bias_ops)
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_mac_convNd(m, x, y: torch.Tensor):
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops)
    m.total_ops += torch.DoubleTensor([int(total_ops)])


@torch.no_grad()
def get_flops_and_params(model, input_shape=(640, 640), gops=True):
    input = torch.randn((1, 3) + input_shape)
    count_conv = count_gops_convNd if gops else count_mac_convNd
    custom_op = {
        torch.nn.Conv2d: count_conv,
        torch.nn.BatchNorm2d: zero_ops,
        Conv2d: count_conv,
        Conv2dStaticSamePadding: count_conv,
        Conv2dDynamicSamePadding: count_conv,
    }
    flops, params = profile(deepcopy(model), inputs=(input,), custom_ops=custom_op, verbose=False)
    return flops, params
