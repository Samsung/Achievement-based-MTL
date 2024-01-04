from typing import Iterator, Tuple

import torch
import torch.nn as nn
from functools import partial

from models.VM import VM
from layers.fpn import FPN, BiFPN, TRIV2BiFPN

from nn.backbones.backbone_factory import backbone_factory
from nn.modules import conv_block_factory, DSConv, MBConv
from core.utils.bbox.bbox import bbox_factory, anchor_generator


# RefineDet CONFIGS
config = {
    '(320, 320)': {
        'min_dim': (320, 320),
        'min_sizes': (32, 64, 128, 256),
        'aspect_ratios': ([2], [2], [2], [2]),
        'feature_maps': ((40, 40), (20, 20), (10, 10), (5, 5)),
        'sizes': ([1], [1], [1], [1]),
        'steps': (8, 16, 32, 64),
        'clip': True,
        'bbox_type': 'SSD',
    },
    '(512, 512)': {
        'min_dim': (512, 512),
        'aspect_ratios': ([2], [2], [2], [2]),
        'feature_maps': ((64, 64), (32, 32), (16, 16), (8, 8)),
        'min_sizes': (32, 64, 128, 256),
        'sizes': ([1], [1], [1], [1]),
        'steps': (8, 16, 32, 64),
        'clip': True,
        'bbox_type': 'SSD',
    },
    '(640, 640)': {
        'min_dim': (640, 640),
        'steps': (8, 16, 32, 64, 128),
        'feature_maps': ((80, 80), (40, 40), (20, 20), (10, 10), (5, 5)),
        'min_sizes': (32, 64, 128, 256, 512),
        'sizes': ([1], [1], [1], [1], [1]),
        'aspect_ratios': ([2], [2], [2], [2], [2]),
        'clip': True,
        'bbox_type': 'SSD',
    },
    '(480, 640)': {
        'min_dim': (480, 640),
        'steps': (8, 16, 32, 64),
        'feature_maps': ((60, 80), (30, 40), (15, 20), (8, 10)),
        'min_sizes': (32, 64, 128, 256),
        'sizes': ([1], [1], [1], [1]),
        'aspect_ratios': ([2], [2], [2], [2]),
        'clip': True,
        'bbox_type': 'SSD',
    }
}


activation_dict = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'swish': nn.SiLU,
    'silu': nn.SiLU,
}

conv_dict = {
    'dsconv': DSConv,
    'mbconv': MBConv
}


class EfficientDet(VM):
    """Vision Multi-Task Model Architecture
    The network is composed of a pretrained backbone network followed by a normal layer
    for each level. The normalized features are given to the feature pyramid network (FPN)
    which combines the features of various levels. Finally, task-specific heads are
    conducted on the outputs of FPN to generate task prediction.

    Args:
        applications: tuple of target applications
        backbone: name of the backbone network
        size: input image size
        num_classes: num of classes of object detection and semantic segmentation
        fpn_type: type of feature pyramid network
        head_type: type of prediction head
        cls_head: type of classification head (softmax for cross entropy loss or sigmoid for focal loss
        normal_size: the number of output channels for normal layers
    """

    def __init__(
            self,
            applications=('detection',),
            backbone='resnet50',
            size=640,
            num_classes=1,
            fpn_type='bifpn',
            head_type='dsconv',
            activation='swish',
            cls_head='sigmoid',
            normal_size=88,
            from_scratch=False,
    ):
        super(EfficientDet, self).__init__(size, applications, num_classes)
        activation = activation_dict[activation]
        self.backbone, in_channels = backbone_factory(backbone, image_size=self.size, from_scratch=from_scratch,
                                                      activation=activation)

        self.cfg = config[str(size)]
        self.cls_head = cls_head
        self.num_classes = self.num_classes - 1 if cls_head == 'sigmoid' else self.num_classes

        levels = 5
        conv_block = conv_dict[head_type.lower()]
        out_channels = [normal_size for _ in range(levels)]
        self.transition = self._build_transitions(in_channels, out_channels, activation=None)
        self._build_fpn(fpn_type, out_channels, normal_size, num_stack=4, conv_block=conv_block, activation=activation)
        self._build_detector(out_channels, num_stack=3, conv_block=conv_block, activation=activation)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
        """
        x, sources = self.backbone.forward(x)
        normal_sources = [normal(x) for x, normal in zip(sources, self.transition)]
        normal_sources += [self.transition[3](sources[-1])]
        normal_sources += [self.transition[4](normal_sources[-1])]
        sources = self.FPN(normal_sources)

        output = dict()
        if 'detection' in self.applications:
            # apply ODM to source layers
            loc, conf = list(), list()
            for x, l, c in zip(sources, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            loc = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], 1)
            output['detection'] = [loc, conf]
        return output

    def _build_fpn(self, fpn_type, in_channels, out_channels, num_stack=4, conv_block=DSConv, activation=nn.ReLU6):
        sizes = self.cfg['feature_maps'][:len(in_channels)]
        fpn_dict = {
            'fpn': partial(FPN, in_channels=in_channels),
            'bifpn': partial(BiFPN, channels=in_channels, sizes=sizes),
            'triv2-bifpn': partial(TRIV2BiFPN, out_channels, sizes),
        }
        fpn_module = fpn_dict[fpn_type]
        self.FPN = nn.Sequential(*[fpn_module(activation=activation) for _ in range(num_stack)])

    def _build_transitions(self, in_channels, out_channels, activation=None, image_sizes=None):
        if image_sizes is None:
            image_sizes = self.cfg['feature_maps']

        normal = [conv_block_factory(in_channel, out_channel, 1, activation=activation, image_size=image_size)
                  for in_channel, out_channel, image_size in zip(in_channels, out_channels, image_sizes)]
        normal += [nn.Sequential(
            conv_block_factory(in_channels[-1], out_channels[-1], 1, activation=activation, image_size=image_sizes[3]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]
        normal += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        return nn.ModuleList(normal)

    def _build_detector(self, in_channels, num_stack=3, conv_block=DSConv, activation=nn.SiLU, mboxes=(3, 3, 3, 3, 3)):
        prior_box = anchor_generator(self.cfg, len(in_channels))
        self.priors = bbox_factory(prior_box)

        self.loc = nn.ModuleList([nn.Sequential(*[
            conv_block(in_channel, mboxes[k] * 4 if i == (num_stack - 1) else in_channel, 3, activation=activation)
            for i in range(num_stack)]) for k, in_channel in enumerate(in_channels)])
        self.conf = nn.ModuleList([nn.Sequential(*[
            conv_block(in_channel, mboxes[k] * self.num_classes if i == num_stack - 1 else in_channel, 3,
                       activation=activation) for i in range(num_stack)]) for k, in_channel in enumerate(in_channels)])

        if self.cls_head == 'sigmoid':  # Initialization for focal loss
            prior_prob = torch.tensor(0.01)
            bias_value = -torch.log((1 - prior_prob) / prior_prob)
            if conv_block == MBConv:
                for layer in self.conf:
                    torch.nn.init.constant_(layer[-1].blocks.proj_conv.normal.bias, bias_value)
            elif conv_block == DSConv:
                for layer in self.loc:
                    layer[-1].pointwise = nn.Conv2d(in_channels[-1], mboxes[-1] * 4, (1, 1))
                for layer in self.conf:
                    layer[-1].pointwise = nn.Conv2d(in_channels[-1], mboxes[-1] * self.num_classes, (1, 1))
                    torch.nn.init.constant_(layer[-1].pointwise.bias, bias_value)

    def get_priors(self):
        return self.priors if hasattr(self, 'priors') else None

    def get_pretrained_params(self) -> Iterator[Tuple[str, nn.Parameter]]:
        return self.backbone.named_parameters()

    def get_from_scratch_params(self) -> Iterator[Tuple[str, nn.Parameter]]:
        named_params = list(self.transition.named_parameters())
        named_params += list(self.FPN.named_parameters())
        if hasattr(self, 'loc'):
            named_params += list(self.loc.named_parameters())
        if hasattr(self, 'conf'):
            named_params += list(self.conf.named_parameters())
        if hasattr(self, 'segment'):
            named_params += list(self.segment.named_parameters())
        if hasattr(self, 'depth'):
            named_params += list(self.depth.named_parameters())
        return named_params

    def get_last_shared_params(self):
        return [param for name, param in self.FPN.named_parameters() if 'weight' in name and 'bn' not in name][-1]

    @staticmethod
    def is_shared(name):
        return any([shared in name for shared in ['backbone', 'transition', 'FPN']])
