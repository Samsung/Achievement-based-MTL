import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .VM import VM
from layers import get_pooling_module
from nn.modules import ConvBNReLU
from nn.backbones.backbone_factory import backbone_factory


class MultiTaskSharedHead(nn.Module):
    def __init__(self, in_channels, normal_size, num_classes, bins, skip, image_size, size, applications=()):
        super(MultiTaskSharedHead, self).__init__()
        self.app = applications
        self.size = size

        self.shared_aspp = get_pooling_module('aspp', in_channels[-1], normal_size, bins, image_size)
        last_channel = normal_size

        if skip:
            self.skip_conv = ConvBNReLU(in_channels[0], skip, 1, activation=nn.ReLU, image_size=image_size)
            self.skip_size = (int(size / 8), int(size / 8)) if not isinstance(size, (list, tuple)) \
                else tuple(int(_size / 8) for _size in size)
            last_channel += skip

        if 'segmentation' in applications:
            self.segmentor = nn.Sequential(
                ConvBNReLU(last_channel, normal_size, 3, activation=nn.ReLU),
                nn.Conv2d(normal_size, num_classes, 1)
            )

        if 'depth' in applications:
            self.depth = nn.Sequential(
                ConvBNReLU(last_channel, normal_size, 3, activation=nn.ReLU),
                nn.Conv2d(normal_size, 1, 1)
            )

        if 'normal' in applications:
            self.surface_normal = nn.Sequential(
                ConvBNReLU(last_channel, normal_size, 3, activation=nn.ReLU),
                nn.Conv2d(normal_size, 3, 1)
            )

    def forward(self, features):
        x = features[-1]
        x = self.shared_aspp(x)

        if hasattr(self, 'skip_conv'):
            x = F.interpolate(x, self.skip_size, mode='bilinear', align_corners=False)
            decoded = self.skip_conv(features[0])
            x = torch.cat([x, decoded], dim=1)

        output = dict()
        if 'depth' in self.app:
            output['depth'] = F.interpolate(self.depth(x), self.size, mode='bilinear', align_corners=False)

        if 'segmentation' in self.app:
            output['segmentation'] = F.interpolate(self.segmentor(x), self.size, mode='bilinear', align_corners=False)

        if 'normal' in self.app:
            normal = F.interpolate(self.surface_normal(x), self.size, mode='bilinear', align_corners=False)
            output['normal'] = F.normalize(normal, dim=1, p=2)
        return output


class MultiTaskHead(nn.Module):
    def __init__(self, in_channels, normal_size, num_classes, bins, skip, image_size, size, applications=()):
        super(MultiTaskHead, self).__init__()
        self.app = applications
        self.size = size

        head_type = partial(DeepLabV3PlusHead, skip=skip) if skip else DeepLabV3Head
        task_head = partial(head_type, in_channels=in_channels, normal_size=normal_size, bins=bins,
                            image_size=image_size, size=size)
        if 'segmentation' in applications:
            self.segmentor = task_head(num_classes=num_classes)

        if 'depth' in applications:
            self.depth = task_head(num_classes=1)

        if 'normal' in applications:
            self.surface_normal = task_head(num_classes=3)

    def forward(self, features):
        output = dict()
        if 'depth' in self.app:
            output['depth'] = self.depth(features)

        if 'segmentation' in self.app:
            output['segmentation'] = self.segmentor(features)

        if 'normal' in self.app:
            normal = self.surface_normal(features)
            output['normal'] = normal / torch.norm(normal, p=2, dim=1, keepdim=True)
        return output


class DeepLabV3Head(nn.Module):
    def __init__(self, in_channels, normal_size, num_classes, bins, image_size, size):
        super(DeepLabV3Head, self).__init__()
        self.size = size

        self.predictor = nn.Sequential(
                get_pooling_module('aspp', in_channels[-1], normal_size, bins, image_size),
                ConvBNReLU(normal_size, normal_size, 3, activation=nn.ReLU),
                nn.Conv2d(normal_size, num_classes, 1)
            )

    def forward(self, features):
        x = features[-1]
        x = self.predictor(x)
        x = F.interpolate(x, self.size, mode='bilinear', align_corners=False)
        return x


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, in_channels, normal_size, num_classes, bins, skip, image_size, size):
        super(DeepLabV3PlusHead, self).__init__()
        self.size = size

        self.aspp = get_pooling_module('aspp', in_channels[-1], normal_size, bins, image_size)
        self.skip_conv = ConvBNReLU(in_channels[0], skip, 1, image_size=image_size)
        self.skip_size = (int(size / 8), int(size / 8)) if not isinstance(size, (list, tuple)) \
            else tuple(int(_size / 8) for _size in size)

        self.predictor = nn.Sequential(
                ConvBNReLU(normal_size + skip, normal_size, 3, activation=nn.ReLU),
                nn.Conv2d(normal_size, num_classes, 1)
            )

    def forward(self, features):
        x = features[-1]
        x = self.aspp(x)

        x = F.interpolate(x, self.skip_size, mode='bilinear', align_corners=False)
        decoded = self.skip_conv(features[0])
        x = torch.cat([x, decoded], dim=1)

        x = self.predictor(x)
        x = F.interpolate(x, self.size, mode='bilinear', align_corners=False)
        return x


class DeepLab(VM):
    r"""DeepLabV3Plus
    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """
    def __init__(self, size, num_classes, backbone, skip_connection, shared_head, normal_size, output_stride=16,
                 from_scratch=False, applications=()):
        super(DeepLab, self).__init__(size, applications, num_classes)

        self.from_scratch = from_scratch
        self.backbone, in_channels = backbone_factory(backbone)

        if output_stride == 8:
            bins = (12, 24, 36)
        elif output_stride == 16:
            bins = (6, 12, 18)
        elif output_stride == 32:
            bins = (6, 12, 18)
        else:
            raise NotImplementedError

        image_size = tuple(int(_size / output_stride) for _size in size) if isinstance(size, (tuple, list)) \
            else size / output_stride

        multi_task_head = MultiTaskSharedHead if shared_head else MultiTaskHead
        self.predict = multi_task_head(in_channels, normal_size, num_classes, bins, skip_connection,
                                       image_size, size, applications)

    def get_pretrained_params(self):
        return self.backbone.named_parameters()

    def get_from_scratch_params(self):
        return self.predict.named_parameters()

    def get_last_shared_params(self):
        if isinstance(self.predict, MultiTaskSharedHead):
            return self.predict.shared_aspp.out_conv[0].weight
        else:
            return [param for name, param in self.backbone.named_parameters()
                    if 'weight' in name and 'bn' not in name][-1]

    @staticmethod
    def is_shared(name):
        return any([shared in name for shared in ['backbone', 'predict.shared', 'predict.skip_conv']])

    def forward(self, x):
        x, features = self.backbone(x)
        output = self.predict(features)
        return output

    def initialize(self, checkpoint):
        if checkpoint:
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] if k[:7] == 'module.' else k

                # for compatibility with legacy
                if 'encoder' in name:
                    name = name.replace('encoder', 'backbone')
                elif 'head.c1_block.conv' in name:
                    name = name.replace('head.c1_block.conv', 'head.c1_block.0')
                elif 'head.c1_block.bn' in name:
                    name = name.replace('head.c1_block.bn', 'head.c1_block.1')

                new_state_dict[name] = v
            self.load_state_dict(new_state_dict, strict=True)
        else:
            print('Initializing weights...')
            self.backbone.get_pretrained_weights(from_scratch=self.from_scratch)
            # ToDo: Initialize the other layers
