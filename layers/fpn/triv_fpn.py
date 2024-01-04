import torch.nn as nn
from functools import partial

from nn.modules import MBConv, DSConv
from layers.module import xavier_init


class TRIV2BiFPN(nn.Module):
    def __init__(self, channels, sizes=(64, 32, 16, 8), conv_type='mbconv', activation=nn.ReLU6):
        super(TRIV2BiFPN, self).__init__()
        self.levels = len(sizes)
        self.size = sizes

        conv_dict = {'dsconv': DSConv, 'mbconv': partial(MBConv, expand_ratio=2)}
        conv_block = conv_dict[conv_type.lower()]
        conv_block = partial(conv_block, in_channels=channels, out_channels=channels, kernel_size=3, stride=2,
                             activation=activation)
        self.upwards = nn.ModuleList([conv_block() for size in sizes[:-1]])
        self.downwards = nn.ModuleList([nn.UpsamplingNearest2d(scale_factor=2) for size in sizes])

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        # top-down
        intermediate = [inputs.pop()]
        for downward, x in zip(self.downwards, inputs[::-1]):
            intermediate.append(x + downward(intermediate[-1]))

        # bottom-up
        outputs = [intermediate.pop()]
        for x, module in zip(intermediate[::-1], self.upwards):
            outputs.append(x + module(outputs[-1]))
        return outputs
