import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, output_padding=None, dilation=1,
                 padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        assert kernel_size == 2 or kernel_size == 3, "Upsample by ConvTranspose2d only supports kernel size of 2 or 3"
        assert stride == 2, "Upsample by ConvTranspose2d only supports stride of 2."
        assert in_channels == out_channels, "in_channels and out_channels must be the same"
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Upsample, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                       in_channels, False, dilation, padding_mode, **factory_kwargs)
        if kernel_size == 2:
            self.output_padding = (0, 0, 0, 0)
            self.weight.data = torch.Tensor([[1.0, 1.0, 1.0, 1.0] for _ in range(in_channels)]).reshape([-1, 1, 2, 2])
        elif kernel_size == 3:
            self._pad = (0, 1, 0, 1)
            self.output_padding = (1, 1, 2, 2)
            self.weight.data = torch.Tensor([[0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]
                                             for _ in range(self.in_channels)]).reshape([-1, 1, 3, 3])
        self.weight.requires_grad = False

    def forward(self, x, output_size=None):
        if hasattr(self, '_pad'):
            x = F.pad(x, self._pad, mode='replicate')
        x = F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, 0, self.groups, self.dilation)
        return x[:, :, self.output_padding[0]:-self.output_padding[2], self.output_padding[1]:-self.output_padding[3]]
