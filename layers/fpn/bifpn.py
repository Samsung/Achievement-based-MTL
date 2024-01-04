import torch
import torch.nn as nn
from functools import partial
from nn.modules import DSConv


class BiFPN(nn.Module):
    def __init__(self, channels, sizes=(80, 40, 20, 10, 5), activation=nn.SiLU):
        super().__init__()
        self.size = sizes
        self.epsilon = 1e-4

        _conv_block = partial(DSConv, kernel_size=3, stride=1, activation=activation)
        self.upwards = nn.ModuleList([_conv_block(in_channels=channel, out_channels=channel)
                                      for channel in channels[:-1]])
        self.downwards = nn.ModuleList([_conv_block(in_channels=channel, out_channels=channel)
                                        for channel in channels[1:]])

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.downsample = nn.MaxPool2d(3, 2, 1)

        levels = len(channels) - 1
        self.weight_upwards = nn.Parameter(torch.ones([levels, 2]), requires_grad=True)
        self.weight_downwards = nn.Parameter(torch.ones([levels, 3]), requires_grad=True)
        self.weight_downwards.data[-1, 1] = 0

        self._weight_upwards = nn.Parameter(torch.ones([levels, 2]), requires_grad=False)
        self._weight_downwards = nn.Parameter(torch.ones([levels, 3]), requires_grad=False)

        self.swish = nn.ModuleList([nn.SiLU(inplace=True) for _ in range(2 * levels)])

    def forward(self, sources):
        if self.training:
            return self._forward_training(sources)
        else:
            return self._forward_eval(sources)

    def _forward_training(self, sources):
        features = [sources[-1]]
        sources = sources[::-1]  # (p7_in, p6_in, p5_in, p4_in, p3_in) 
        for i in range(len(sources) - 1):
            _w = torch.relu(self.weight_upwards[i])
            _w = _w / (torch.sum(_w, dim=0) + self.epsilon)
            feature = self.upwards[i](self.swish[i](_w[0] * sources[i + 1] + _w[1] * self.upsample(features[-1])))
            features.append(feature)
        features = features[1:]

        outputs = [features[-1]]
        sources = sources[::-1]  # (p3_in, p4_in, p5_in, p6_in, p7_in)
        features = features[::-1] + [0]  # (p3_out, p4_up, p5_up, p6_up, 0)
        for i, feature in enumerate(features[1:]):
            _w = torch.relu(self.weight_downwards[i])
            _w = _w / (torch.sum(_w, dim=0) + self.epsilon)
            feature = self.downwards[i](
                self.swish[i + 4](_w[0] * sources[i + 1] +
                                  _w[1] * features[i + 1] +
                                  _w[2] * self.downsample(self.stochastic_depth(outputs[-1]))))
            outputs.append(feature)
        return outputs

    def _forward_eval(self, sources):
        features = [sources[-1]]
        sources = sources[::-1]  # (p7_in, p6_in, p5_in, p4_in, p3_in)
        for i in range(len(sources) - 1):
            _w = self._weight_upwards[i]
            feature = self.upwards[i](self.swish[i](_w[0] * sources[i + 1] + _w[1] * self.upsample(features[-1])))
            features.append(feature)
        features = features[1:]

        outputs = [features[-1]]
        sources = sources[::-1]  # (p3_in, p4_in, p5_in, p6_in, p7_in)
        features = features[::-1] + [0]  # (p3_out, p4_up, p5_up, p6_up, 0)
        for i, feature in enumerate(features[1:]):
            _w = self._weight_downwards[i]
            feature = self.downwards[i](
                self.swish[i + 4](_w[0] * sources[i + 1] +
                                  _w[1] * features[i + 1] +
                                  _w[2] * self.downsample(outputs[-1])))
            outputs.append(feature)
        return outputs

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:  # eval
            self._compute_eval_weights()

    def _compute_eval_weights(self):
        for i in range(self.weight_upwards.shape[0]):
            _w = torch.relu(self.weight_upwards[i])
            self._weight_upwards[i] = _w / (torch.sum(_w, dim=0) + self.epsilon)

        for i in range(self.weight_downwards.shape[0]):
            _w = torch.relu(self.weight_downwards[i])
            self._weight_downwards[i] = _w / (torch.sum(_w, dim=0) + self.epsilon)
