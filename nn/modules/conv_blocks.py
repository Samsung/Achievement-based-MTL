from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import StochasticDepth, SqueezeExcitation, FrozenBatchNorm2d

from functools import partial

from .conv2d import get_conv2d, Conv2d
from typing import Union, Tuple, Type, Callable
_size_2_int = Union[int, Tuple[int, int]]


def conv_block_factory(
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_int = 1,
        stride: _size_2_int = 1,
        padding: _size_2_int = None,
        dilation: int = 1,
        groups: int = 1,
        transposed: bool = False,
        output_padding: _size_2_int = 0,
        norm: Type[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation: nn.Module = None,
        image_size: _size_2_int = None,
        tf_pad: bool = False) -> ConvNormActivation:
    if isinstance(norm, (nn.BatchNorm2d, FrozenBatchNorm2d)) or \
            (not isinstance(norm, nn.Module) and issubclass(norm, (nn.BatchNorm2d, FrozenBatchNorm2d))):
        if activation is not None and isinstance(activation, (nn.ReLU, nn.ReLU6)):
            return ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, transposed,
                              output_padding, norm, activation=activation, image_size=image_size, tf_pad=tf_pad)
        else:
            return ConvBN(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, transposed,
                          output_padding, norm, activation=activation, image_size=image_size, tf_pad=tf_pad)
    else:
        return ConvNormActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, transposed,
                                  output_padding, norm, activation=activation, image_size=image_size, tf_pad=tf_pad)


class ConvNormActivation(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_int = 1,
                 stride: _size_2_int = 1,
                 padding: _size_2_int = None,
                 dilation: int = 1,
                 groups: int = 1,
                 transposed: bool = False,
                 output_padding: _size_2_int = 0,
                 norm: Union[Type[Callable[..., nn.Module]], nn.Module] = nn.BatchNorm2d,
                 activation: Union[Type[Callable[..., nn.Module]], nn.Module] = None,
                 image_size: _size_2_int = None,
                 tf_pad: bool = False,
                 conv_factory: Callable[..., Conv2d] = get_conv2d) -> None:
        super(ConvNormActivation, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = (int((k - 1) / 2) for k in kernel_size) if padding is None else padding
        conv = partial(torch.nn.ConvTranspose2d, output_padding=output_padding) if transposed \
            else conv_factory(image_size, tf_pad)
        self.add_module('conv', conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=False))
        self.add_module('normal', norm if isinstance(norm, nn.Module) else norm(out_channels))
        if activation is not None:
            self.add_module('activation', activation if isinstance(activation, nn.Module) else activation(inplace=True))


class ConvBN(ConvNormActivation):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_int = 1,
                 stride: _size_2_int = 1,
                 padding: _size_2_int = None,
                 dilation: int = 1,
                 groups: int = 1,
                 transposed: bool = False,
                 output_padding: _size_2_int = 0,
                 norm: Union[Type[Callable[..., nn.Module]], nn.Module] = nn.BatchNorm2d,
                 activation: Union[Type[Callable[..., nn.Module]], nn.Module] = None,
                 image_size: _size_2_int = None,
                 tf_pad: bool = False) -> None:
        assert isinstance(norm, (nn.BatchNorm2d, FrozenBatchNorm2d)) or \
               (not isinstance(norm, nn.Module) and issubclass(norm, (nn.BatchNorm2d, FrozenBatchNorm2d))), \
               "Norm of ConvBN should be one of nn.BatchNorm2d or FrozenBatchNorm2d"
        super(ConvBN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                     transposed, output_padding, norm, activation, image_size, tf_pad)

    def fold_bn(self):
        if hasattr(self, 'normal'):  # if ConvBN has BN
            conv, bn = self.conv, self.normal
            scale = bn.weight / (bn.running_var + bn.eps).sqrt()
            conv.weight.data = conv.weight * scale.reshape(-1, 1, 1, 1)
            conv.bias = nn.Parameter(bn.bias + (conv.bias if conv.bias else 0 - bn.running_mean) * scale)
            delattr(self, 'normal')


class ConvBNReLU(ConvBN):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_int = 1,
                 stride: _size_2_int = 1,
                 padding: _size_2_int = None,
                 dilation: int = 1,
                 groups: int = 1,
                 transposed: bool = False,
                 output_padding: _size_2_int = 0,
                 norm: Union[Type[Callable[..., nn.Module]], nn.Module] = nn.BatchNorm2d,
                 activation: Union[Type[Callable[..., nn.Module]], nn.Module] = nn.ReLU6,
                 image_size: _size_2_int = None,
                 tf_pad: bool = False) -> None:
        assert isinstance(activation, (nn.ReLU, nn.ReLU6)) or \
               (not isinstance(activation, nn.Module) and issubclass(activation, (nn.ReLU, nn.ReLU6))), \
               "Activation of ConvBNReLU should be one of nn.ReLU() or nn.ReLU6()"
        super(ConvBNReLU, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                         transposed, output_padding, norm, activation, image_size, tf_pad)


class DSConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_int = 1,
                 stride: _size_2_int = 1,
                 padding: _size_2_int = None,
                 dilation: int = 1,
                 norm: Union[Type[Callable[..., nn.Module]], nn.Module] = nn.BatchNorm2d,
                 activation: Union[Type[Callable[..., nn.Module]], nn.Module] = None,
                 image_size: _size_2_int = None,
                 tf_pad: bool = False,
                 get_conv_block: Callable[..., nn.Sequential] = conv_block_factory):
        super(DSConv, self).__init__()
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        conv = get_conv2d(image_size=image_size, tf_pad=tf_pad)
        padding = (int((k - 1) / 2) for k in kernel_size) if padding is None else padding
        self.depthwise = conv(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, bias=False)
        self.pointwise = get_conv_block(in_channels, out_channels, kernel_size=1, groups=1,
                                        norm=norm, activation=activation, image_size=image_size, tf_pad=tf_pad)

    def forward(self, x: Tensor) -> Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class MBConv(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        kernel_size (int): Size of the convolving kernel
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        expand_ratio (int): Expand ratio of MBConv
        se_ratio (string, optional): SE ratio of the SE module.
                                     if the SE module is not used in MBConv, se_ratio should be 'None'
        id_skip: if 'True', MBConv has skip_connect
        image_size (int or tuple, optional): Input image size
        tf_pad (bool, optional): if 'True', Conv used TF style padding
        activation ('string'): Activation type used in MBConv.

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 expand_ratio: int = 2,
                 expand_conv: bool = True,
                 se_ratio: Union[float, None] = None,
                 id_skip: bool = True,
                 transposed: bool = False,
                 output_padding: _size_2_int = 0,
                 image_size: _size_2_int = None,
                 tf_pad: bool = False,
                 activation: Callable[..., nn.Module] = nn.SiLU,
                 se_activation: nn.Module = nn.SiLU,
                 get_conv_block: Callable[..., ConvNormActivation] = conv_block_factory,
                 stochastic_depth_prob: float = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.se_ratio = se_ratio
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.id_skip = id_skip and stride == 1 and in_channels == out_channels  # skip connection and drop connect
        self.expand_ratio = expand_ratio
        self._expand_conv = expand_conv
        self.image_size = image_size
        self.tf_pad = tf_pad
        self.se_activation = se_activation

        # Get static or dynamic convolution depending on image size
        _ConvNormAct = partial(get_conv_block, image_size=image_size, tf_pad=tf_pad)
        _Conv2d = get_conv2d(image_size=image_size, tf_pad=tf_pad)

        # Expansion phase
        self.blocks = nn.Sequential()
        inp = in_channels  # number of input channels
        oup = in_channels * expand_ratio  # number of output channels
        if expand_conv and expand_ratio != 1:
            self.blocks.add_module('expand_conv', _ConvNormAct(inp, oup, 1, activation=activation))
            inp = oup

        # Depthwise convolution phase
        k = kernel_size
        s = stride
        group = oup if expand_conv else 1

        self.blocks.add_module('central_conv', _ConvNormAct(inp, oup, k, groups=group, stride=s, transposed=transposed,
                                                            output_padding=output_padding, activation=activation))

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.blocks.add_module('se', SqueezeExcitation(oup, num_squeezed_channels, activation=se_activation))

        # Output phase
        final_oup = out_channels
        if expand_conv or expand_ratio != 1:
            self.blocks.add_module('proj_conv', _ConvNormAct(oup, final_oup, 1))
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.blocks(inputs)

        # Skip connection and drop connect
        if self.id_skip:
            x = self.stochastic_depth(x)
            x += inputs
        return x
