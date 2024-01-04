import torch.nn as nn
from functools import partial


def get_activation(activation: str, inplace: bool = True, get_instance: bool = True):
    """
    Sets activation function:
        'relu'             :standard relu           (for mobile devices and npu)
        'hswish'           :hard swish              (for mobile devices and npu)
        'memory_efficient' :memory efficient swish  (for training)
        'swish'            :standard swish          (for export-inference only)
    """

    act_dict = {
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'hswish': nn.Hardswish,
        'swish': nn.SiLU,
        'sigmoid': nn.Sigmoid,
        'none': nn.Identity,
    }
    activation = activation.lower()
    assert activation in act_dict, 'activation name %s is invalid' % activation
    activation = partial(act_dict[activation], inplace=inplace)
    return activation() if get_instance else activation
