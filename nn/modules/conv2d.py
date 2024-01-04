import math
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from functools import partial
from typing import Union, Tuple, Callable

_size_2_int = Union[int, Tuple[int, int]]


def get_conv2d(image_size: _size_2_int = None, tf_pad: bool = False) -> Callable[..., nn.Conv2d]:
    """
    Chooses static padding if you have specified an image size, and dynamic padding otherwise.
    Static padding is necessary for ONNX exporting of models.
    """
    if tf_pad:
        return Conv2dDynamicSamePadding if image_size is None \
            else partial(Conv2dStaticSamePadding, image_size=image_size)
    else:
        return nn.Conv2d


class Conv2d(nn.Conv2d):
    """
    PyTorch Style Padding
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_int,
        stride: _size_2_int = 1,
        padding: Union[None, _size_2_int] = 0,
        dilation: _size_2_int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
    ) -> None:
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        padding = (int((k - 1) / 2) for k in kernel_size) if padding is None else padding
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, '_pad'):
            x = self._pad(x)
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dDynamicSamePadding(Conv2d):
    """
    2D Convolutions like TensorFlow, for a dynamic image size
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_int,
        stride: _size_2_int = 1,
        padding: Union[None, _size_2_int] = 0,
        dilation: _size_2_int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def _pad(self, x: Tensor) -> Tensor:
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


class Conv2dStaticSamePadding(Conv2d):
    """
    2D Convolutions like TensorFlow, for a fixed image size
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_int,
        stride: _size_2_int = 1,
        padding: Union[None, _size_2_int] = 0,
        dilation: _size_2_int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        image_size=None,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if isinstance(image_size, (tuple, list)) else (int(image_size), int(image_size))
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        self.padding = (pad_w // 2, pad_h // 2)
        if pad_w % 2 or pad_h % 2:
            self._pad = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
            self.padding = (0, 0)
