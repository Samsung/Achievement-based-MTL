from typing import Optional, Union, Tuple, Callable
from functools import partial
from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor
from . import conv_block_factory
_size_2_int = Union[int, Tuple[int, int]]


class _ResidualBlock(nn.Module, ABC):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: _size_2_int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        image_size: _size_2_int = None,
        tf_pad=False,
    ) -> None:
        super(_ResidualBlock, self).__init__()

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes
        self.groups = groups
        self.base_width = base_width
        self.dilation = dilation
        self.norm_layer = norm_layer
        self.image_size = image_size
        self.tf_pad = tf_pad

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class BasicBlock(_ResidualBlock):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Sequential] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
        image_size=None,
        tf_pad=False,
        get_conv_block=conv_block_factory
    ) -> None:
        super().__init__(inplanes, planes, stride, groups, base_width, dilation, norm_layer, image_size, tf_pad)
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        _ConvNormAct = partial(get_conv_block, norm=norm_layer, image_size=image_size, tf_pad=tf_pad)

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _ConvNormAct(inplanes, planes, 3, stride=stride, activation=nn.ReLU)
        self.conv2 = _ConvNormAct(planes, planes, 3)
        self.relu = nn.ReLU(inplace=True)

        # downsample is re-organized based on the original implementation
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L239-L243
        self.downsample = _ConvNormAct(inplanes, planes * self.expansion, 1, stride) if downsample else None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(_ResidualBlock):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Sequential] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        image_size=None,
        tf_pad=False,
        get_conv_block=conv_block_factory,
    ) -> None:
        super().__init__(inplanes, planes, stride, groups, base_width, dilation, norm_layer, image_size, tf_pad)

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        _ConvNormAct = partial(get_conv_block, image_size=image_size, tf_pad=tf_pad, norm=norm_layer)

        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = _ConvNormAct(inplanes, width, kernel_size=1, stride=1, activation=nn.ReLU)
        self.conv2 = _ConvNormAct(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups,
                                  dilation=dilation, activation=nn.ReLU)
        self.conv3 = _ConvNormAct(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # downsample is re-organized based on the original implementation
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L239-L243
        self.downsample = _ConvNormAct(inplanes, planes * self.expansion, 1, stride) if downsample else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)  # ConvBNReLU
        out = self.conv2(out)  # ConvBNReLU
        out = self.conv3(out)  # ConvBN

        if self.downsample is not None:
            identity = self.downsample(x)
        assert out.size() == identity.size(), "out.size: {}, identity.size(): {}".format(out.size(), identity.size())
        out += identity
        out = self.relu(out)
        return out
