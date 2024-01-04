from typing import overload, Type, Callable

from torch import nn
from .conv_blocks import conv_block_factory, MBConv, DSConv, ConvNormActivation, _size_2_int
from .residual_blocks import BasicBlock, Bottleneck


@overload
def conv_block_factory2(
        conv_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_int = 1,
        stride: _size_2_int = 1,
        padding: _size_2_int = None,
        dilation: int = 1,
        groups: int = 1,
        norm: Type[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation: nn.Module = None,
        image_size: _size_2_int = None,
        tf_pad: bool = False) -> ConvNormActivation:
    pass


def conv_block_factory2(conv_type: str = 'ConvNorm', *args, **kwargs):
    lower_conv_type = conv_type.lower()
    conv_dict = {
        'conv': conv_block_factory,
        'dsconv': DSConv,
        'mbconv': MBConv,
        'basicblock': BasicBlock,
        'bottleneck': Bottleneck,
    }
    assert lower_conv_type in conv_dict, f'{conv_type} is not supported yet'
    return conv_dict[conv_type](*args, **kwargs)
