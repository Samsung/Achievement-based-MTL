from typing import Dict, Type, Callable
from torch import nn

from torchvision.ops import Conv2dNormActivation
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.models.efficientnet import MBConv as NNMBConv
from torchvision.models.efficientnet import FusedMBConv
from ..modules import conv_block_factory, MBConv

convert_dict: Dict[Type, Type] = {
    Conv2dNormActivation: conv_block_factory,
    NNMBConv: MBConv,
    FusedMBConv: MBConv,
    InvertedResidual: MBConv,
}


def convert_fn(module: nn.Module, _convert_list: Dict[Type, Type], **kwargs) -> nn.Module:
    conv = None
    if type(module) == Conv2dNormActivation:
        _conv, _norm, _act = module[0], module[1], module[2]
        _act = kwargs.pop('activation', _act)
        conv = conv_block_factory(in_channels=_conv.in_channels,
                                  out_channels=_conv.out_channels,
                                  kernel_size=_conv.kernel_size,
                                  stride=_conv.stride,
                                  dilation=_conv.dilation,
                                  groups=_conv.groups,
                                  norm=_norm,
                                  activation=_act)
    elif type(module) == InvertedResidual:
        conv = MBConv(kernel_size=3,
                      in_channels=module.conv[0][0].in_channels,
                      out_channels=module.out_channels,
                      stride=module.stride,
                      expand_ratio=int(module.conv[0][0].out_channels / module.conv[0][0].in_channels),
                      activation=nn.ReLU6)
    elif type(module) == NNMBConv:
        main_conv = None
        for mod in module.block.modules():
            if isinstance(mod, nn.Conv2d):
                if mod.in_channels == mod.groups:
                    main_conv = mod
        main_conv = main_conv if main_conv else module.block[0][0]

        # set arguments
        activation = kwargs['activation'] if 'activation' in kwargs else nn.SiLU
        se_ratio = kwargs['se_ratio'] if 'se_ratio' in kwargs else 0.25
        se_activation = kwargs['se_activation'] if 'se_activation' in kwargs else nn.SiLU

        conv = MBConv(
            in_channels=module.block[0][0].in_channels,
            out_channels=module.block[-1][0].out_channels,
            kernel_size=main_conv.kernel_size[0],
            stride=main_conv.stride[0],
            expand_ratio=int(module.block[0][0].out_channels / module.block[0][0].in_channels),
            expand_conv=True,
            activation=activation,
            se_ratio=se_ratio,
            se_activation=se_activation,
            id_skip=module.use_res_connect,
            stochastic_depth_prob=module.stochastic_depth.p,
        )
    elif type(module) == FusedMBConv:
        main_conv = None
        for mod in module.block.modules():
            if isinstance(mod, nn.Conv2d):
                if mod.in_channels == mod.groups:
                    main_conv = mod
        main_conv = main_conv if main_conv else module.block[0][0]

        # set arguments
        activation = kwargs['activation'] if 'activation' in kwargs else nn.SiLU
        se_ratio = None
        se_activation = kwargs['se_activation'] if 'se_activation' in kwargs else nn.SiLU

        conv = MBConv(
            in_channels=module.block[0][0].in_channels,
            out_channels=module.block[-1][0].out_channels,
            kernel_size=main_conv.kernel_size[0],
            stride=main_conv.stride[0],
            expand_ratio=int(module.block[0][0].out_channels / module.block[0][0].in_channels),
            expand_conv=False,
            activation=activation,
            se_ratio=se_ratio,
            se_activation=se_activation,
            id_skip=module.use_res_connect,
            stochastic_depth_prob=module.stochastic_depth.p,
        )
    return conv


def convert_model(model: nn.Module, _convert_dict: Dict[Type, Type] = convert_dict,
                  _convert_fn: Callable[..., nn.Module] = convert_fn, **kwargs) -> None:
    for name, module in model.named_children():
        if type(module) in _convert_dict:
            setattr(model, name, convert_fn(module, _convert_dict, **kwargs))
        else:
            convert_model(module, _convert_dict, _convert_fn, **kwargs)
