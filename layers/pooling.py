import torch
from torch import nn
from torch.nn import functional as F
from nn.modules import ConvBNReLU  # ToDo: this will be replaced with get_conv_block


class ASPP(nn.Module):
    """ Atrous Spatial Pyramid Pooling Module """
    """
              - ConvBNReLU ------------------------------------------>
              - Atrous DSConv ---------------------------------------> 
              - Atrous DSConv --------------------------------------->
    origin -> - Atrous DSConv ---------------------------------------> concat -> conv to output_ch
              - Atrous DSConv ---------------------------------------> 
              - Adaptive Avg. pooling -> Conv -> resize to original ->
    """
    def __init__(self, in_channel, out_channel, bins, image_size):
        super().__init__()
        self.image_size = (int(image_size), int(image_size)) if not isinstance(image_size, (list, tuple)) \
            else image_size

        acc_channel = out_channel
        features = [ConvBNReLU(in_channel, out_channel, 1, image_size=image_size)]
        for _bin in bins:
            features.append(ConvBNReLU(in_channel, out_channel, 3, padding=_bin, dilation=_bin))
            acc_channel += out_channel
        features.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                      ConvBNReLU(in_channel, out_channel, 1, image_size=1),
                                      nn.Upsample(size=self.image_size, mode='bilinear', align_corners=False)))
        acc_channel += out_channel
        self.features = nn.ModuleList(features)

        self.out_conv = ConvBNReLU(acc_channel, out_channel, 1, image_size=image_size)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        x = torch.cat([feature(x) for feature in self.features], dim=1)
        x = self.out_conv(x)
        x = self.dropout(x)
        return x


class PPM(nn.Module):
    """ Pyramid Pooling Module """
    """
              - pool feature size : bin[0] -> pwconv ch x 1/4 -> resize to original 
              - pool feature size : bin[1] -> pwconv ch x 1/4 -> resize to original
    origin -> - pool feature size : bin[2] -> pwconv ch x 1/4 -> resize to original -> concat -> pwconv to output_ch
              - pool feature size : bin[3] -> pwconv ch x 1/4 -> resize to original 
              - ---------------------------- origin ----------------------------- -
    """
    # ToDo: Conv layers will be replaced with get_conv_2d
    def __init__(self, in_channel, out_channel, bins, image_size):
        super(PPM, self).__init__()
        if bins is None:
            bins = (1, 2, 4, 8)
        self.image_size = image_size
        pyramid_level = len(bins)
        reduction_dim = int(in_channel / pyramid_level)
        features = [nn.Sequential(nn.AdaptiveAvgPool2d(bin),
                                  nn.Conv2d(in_channel, reduction_dim, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(reduction_dim),
                                  nn.ReLU(inplace=True)) for bin in bins]
        self.features = nn.ModuleList(features)
        self.out_conv = nn.Sequential(nn.Conv2d(in_channel * 2, out_channel, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_channel),
                                      nn.ReLU(inplace=True))

    def forward(self, x):
        out = [x] + [F.interpolate(f(x), self.image_size, mode='bilinear', align_corners=True) for f in self.features]
        out = torch.cat(out, 1)
        out = self.out_conv(out)

        return out


class FlexPPMSum(nn.Module):
    """ Pyramid Pooling Module """
    """
              - pool x 1/8 -> pwconv out ch -> resize to original 
              - pool x 1/4 -> pwconv out ch -> resize to original
    origin -> - pool x 1/2 -> pwconv out ch -> resize to original -> sum ->  
              - pool x 1/1 -> pwconv out ch -> resize to original
    """
    # ToDo: Conv layers will be replaced with get_conv_2d
    def __init__(self, in_channel, out_channel, bins, image_size):
        super(FlexPPMSum, self).__init__()
        if bins is None:
            bins = (1, 2, 4, 8)
        self.image_size = image_size
        features = [nn.Sequential(nn.AvgPool2d(ratio, ceil_mode=True),
                                  nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  nn.Upsample(size=image_size, mode='bilinear', align_corners=False))
                    for ratio in bins]
        self.features = nn.ModuleList(features)
        self.observer = nn.Identity()  # it will be replaced with observer for QAT

    def forward(self, x):
        return self.observer(sum([f(x) for f in self.features]))


class SppYolo(nn.Module):
    """ Which is used in YOLO """
    """
                                      -----original size feature--
                                      - pool kernel=5 , stride=1 -
    origin -> pwconv to input_ch/2 -> - pool kernel=9 , stride=1 --> concat -> pwconv to output_ch 
                                      - pool kernel=13, stride=1 -
    
    """
    # ToDo: Conv layers will be replaced with get_conv_2d
    def __init__(self, in_channel, out_channel, bins, image_size):
        super(SppYolo, self).__init__()
        if bins is None:
            bins = (5, 9, 13)
        features = [nn.Sequential(nn.MaxPool2d(kernel_size=k, stride=1, padding=int(k / 2)))
                    for k in bins]
        self.features = nn.ModuleList(features)
        self.in_conv = nn.Sequential(nn.Conv2d(in_channel, int(in_channel / 2), kernel_size=1, bias=False),
                                     nn.BatchNorm2d(int(in_channel / 2)),
                                     nn.LeakyReLU(inplace=True))
        self.out_conv = nn.Sequential(nn.Conv2d(in_channel * 2, out_channel, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_channel),
                                      nn.LeakyReLU(inplace=True))

    def forward(self, x):
        out = self.in_conv(x)
        out = [out] + [f(out) for f in self.features]
        out = torch.cat(out, 1)
        out = self.out_conv(out)

        return out
