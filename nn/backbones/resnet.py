from typing import Type, Union, List, Optional, Tuple
from collections import OrderedDict

import torch
from torch import Tensor

from torchvision.models import ResNet as nnResNet
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, \
    ResNet101_Weights, ResNet152_Weights
from ..modules import conv_block_factory, BasicBlock, Bottleneck  # custom blocks


def get_state_dict(model_name: str):
    from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
    model_dict = {
        'resnet18': resnet18, 'resnet34': resnet34,
        'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152,
    }
    return model_dict[model_name](model_config_dict[model_name]['weights']).state_dict()


model_config_dict = {
    'resnet18': {'arg_dict': [BasicBlock, [2, 2, 2, 2]], 'weights': ResNet18_Weights.IMAGENET1K_V1},
    'resnet34': {'arg_dict': [BasicBlock, [3, 4, 6, 3]], 'weights': ResNet34_Weights.IMAGENET1K_V1},
    'resnet50': {'arg_dict': [Bottleneck, [3, 4, 6, 3]], 'weights': ResNet50_Weights.IMAGENET1K_V2},
    'resnet101': {'arg_dict': [Bottleneck, [3, 4, 23, 3]], 'weights': ResNet101_Weights.IMAGENET1K_V2},
    'resnet152': {'arg_dict': [Bottleneck, [3, 8, 36, 3]], 'weights': ResNet152_Weights.IMAGENET1K_V2},
}


class ResNet(nnResNet):
    def __init__(
        self,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation)
        self.conv_stem = conv_block_factory(3, 64, kernel_size=7, stride=2, norm=self.bn1, activation=self.relu)
        delattr(self, 'bn1')
        delattr(self, 'relu')

        self.__dict__['_modules'] = \
            OrderedDict({'conv_stem' if k == 'conv1' else k: v for k, v in self.__dict__['_modules'].items()})

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv_stem(x)  # ConvBNReLU
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x

    def load_state_dict(self, state_dict, strict: bool = True):
        super(ResNet, self).load_state_dict(state_dict, strict)


class ResNetBackbone(ResNet):
    def __init__(
        self,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation)
        delattr(self, 'fc')
        delattr(self, 'avgpool')

        extraction_points = [self.layer2, self.layer3, self.layer4]
        self.in_channels = [layer[-1].conv3[0].out_channels for layer in extraction_points]

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        features = []

        x = self.conv_stem(x)  # ConvBNReLU
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        if hasattr(self, 'extra'):
            x = self.extra(x)
            features.append(x)
        return x, features
