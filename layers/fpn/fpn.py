import torch.nn as nn

class FPN(nn.Module):
    def __init__(self, in_channels, use_bn):
        super(FPN, self).__init__()
        in_channel = 256
        self.fpn = []
        if use_bn:
            for k, v in enumerate(in_channels):
                self.fpn.append([])
                self.fpn[k] = [nn.Conv2d(in_channels[k], in_channel, 3, padding=1, bias=False),
                               nn.BatchNorm2d(in_channel),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
                               nn.BatchNorm2d(in_channel)]

                if k != len(in_channels) - 1:
                    self.fpn[k] += [nn.ConvTranspose2d(in_channel, in_channel, 2, 2, bias=False),
                                    nn.BatchNorm2d(in_channel)]
                self.fpn[k] += [nn.ReLU(inplace=True),
                                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
                                nn.BatchNorm2d(in_channel),
                                nn.ReLU(inplace=True)]
                self.fpn[k] = nn.ModuleList(self.fpn[k])
        else:
            for k, v in enumerate(in_channels):
                self.fpn.append([])
                self.fpn[k] = [nn.Conv2d(in_channels[k], in_channel, 3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(in_channel, in_channel, 3, padding=1)]
                if k != len(in_channels) - 1:
                    self.fpn[k] += [nn.ConvTranspose2d(in_channel, in_channel, 2, 2)]
                self.fpn[k] += [nn.ReLU(inplace=True),
                                nn.Conv2d(in_channel, in_channel, 3, padding=1),
                                nn.ReLU(inplace=True)]
                self.fpn[k] = nn.ModuleList(self.fpn[k])
        self.fpn = nn.ModuleList(self.fpn)

        self.out_channels = [in_channel, in_channel, in_channel, in_channel]

    def forward(self, inputs):
        sources = inputs
        sources.reverse()

        FPN_source = list()
        for v, layers in zip(sources, list(reversed(self.fpn))):
            s = v
            for layer in layers:
                if isinstance(layer, nn.ConvTranspose2d):
                    u = FPN_source[-1]
                    u = layer(u)
                    s += u
                else:
                    s = layer(s)
            FPN_source.append(s)

        FPN_source.reverse()
        sources = FPN_source

        return sources
