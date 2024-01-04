import math
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from functools import partial
from typing import Union, Tuple, Callable

_size_2_int = Union[int, Tuple[int, int]]


def get_max_pool2d(image_size: _size_2_int = None, tf_pad: bool = False) -> Callable[..., nn.MaxPool2d]:
    """
    Chooses static padding if you have specified an image size, and dynamic padding otherwise.
    Static padding is necessary for ONNX exporting of models.
    """
    if tf_pad:
        return MaxPool2dDynamicSamePadding if image_size is None \
            else partial(MaxPool2dStaticSamePadding, image_size=image_size)
    else:
        return MaxPool2d


class MaxPool2d(nn.MaxPool2d):
    """
    PyTorch Style Padding
    """
    def __init__(
        self,
        kernel_size: _size_2_int,
        stride: _size_2_int = 1,
        padding: Union[None, _size_2_int] = 0,
        dilation: _size_2_int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        assert len(kernel_size) == 2, 'A kernel_size type should be "int" or "tuple(int, int)".'
        assert len(stride) == 2, 'A stride type should be "int" or "tuple(int, int)".'
        assert len(dilation) == 2, 'A dilation type should be "int" or "tuple(int, int)".'
        padding = (int((k - 1) / 2) for k in kernel_size) if padding is None else padding
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    @staticmethod
    def _get_pad(i_size, k_size, s_size, d_size):
        oh, ow = math.ceil(i_size[0] / s_size[0]), math.ceil(i_size[1] / s_size[1])
        pad_h = max((oh - 1) * s_size[0] + (k_size[0] - 1) * d_size[0] + 1 - i_size[0], 0)
        pad_w = max((ow - 1) * s_size[1] + (k_size[1] - 1) * d_size[1] + 1 - i_size[1], 0)
        return pad_h, pad_w

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, '_pad'):
            x = self._pad(x)
        return F.max_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices)


class MaxPool2dDynamicSamePadding(MaxPool2d):
    """
    2D MaxPooling like TensorFlow, for a dynamic image size
    """

    def __init__(
        self,
        kernel_size: _size_2_int,
        stride: _size_2_int = 1,
        padding: Union[None, _size_2_int] = 0,
        dilation: _size_2_int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__(kernel_size, stride, 0, dilation, return_indices, ceil_mode)

    def _pad(self, x: Tensor) -> Tensor:
        pad_h, pad_w = self._get_pad(x.size()[-2:], self.kernel_size, self.stride, self.dilation)
        if pad_h > 0 or pad_w > 0:
            return F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


class MaxPool2dStaticSamePadding(MaxPool2d):
    """
    2D MaxPooling like TensorFlow, for a fixed image size
    """

    def __init__(
        self,
        kernel_size: _size_2_int,
        stride: _size_2_int = 1,
        padding: Union[None, _size_2_int] = 0,
        dilation: _size_2_int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        image_size: _size_2_int = None,
    ) -> None:
        super().__init__(kernel_size, stride, 0, dilation, return_indices, ceil_mode)  

        # Calculate padding based on image size and save it
        assert image_size is not None
        pad_h, pad_w = self._get_pad(_pair(image_size), self.kernel_size, self.stride, self.dilation)
        self.padding = (pad_w // 2, pad_h // 2)
        if pad_w % 2 or pad_h % 2:
            self._pad = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
            self.padding = (0, 0)
