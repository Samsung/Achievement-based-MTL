import torch
import torch.nn as nn
from typing import Sequence, Optional, Union, Callable, Any, Tuple, List

from torchvision.models.efficientnet import EfficientNet as NNEfficientNet
from torchvision.models.efficientnet import _efficientnet_conf, MBConvConfig, FusedMBConvConfig
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, \
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, \
    EfficientNet_B7_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights

from .utils import convert_model


model_config_dict = {
    'efficientnet_b0': {'arg_dict': _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0),
                        'extraction_points': (3, 5, 6),
                        'weights': EfficientNet_B0_Weights.IMAGENET1K_V1},
    'efficientnet_b1': {'arg_dict': _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1),
                        'extraction_points': (3, 5, 7),
                        'weights': EfficientNet_B1_Weights.IMAGENET1K_V2},
    'efficientnet_b2': {'arg_dict': _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2),
                        'weights': EfficientNet_B2_Weights.IMAGENET1K_V1},
    'efficientnet_b3': {'arg_dict': _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4),
                        'weights': EfficientNet_B3_Weights.IMAGENET1K_V1},
    'efficientnet_b4': {'arg_dict': _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8),
                        'weights': EfficientNet_B4_Weights.IMAGENET1K_V1},
    'efficientnet_b5': {'arg_dict': _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2),
                        'weights': EfficientNet_B5_Weights.IMAGENET1K_V1},
    'efficientnet_b6': {'arg_dict': _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6),
                        'weights': EfficientNet_B6_Weights.IMAGENET1K_V1},
    'efficientnet_b7': {'arg_dict': _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1),
                        'weights': EfficientNet_B7_Weights.IMAGENET1K_V1},
    'efficientnet_v2_s': {'arg_dict': _efficientnet_conf("efficientnet_v2_s"),
                          'extraction_points': (3, 5, 6),
                          'weights': EfficientNet_V2_S_Weights.IMAGENET1K_V1},
    'efficientnet_v2_m': {'arg_dict': _efficientnet_conf("efficientnet_v2_m"),
                          'weights': EfficientNet_V2_M_Weights.IMAGENET1K_V1},
    'efficientnet_v2_l': {'arg_dict': _efficientnet_conf("efficientnet_v2_l"),
                          'weights': EfficientNet_V2_L_Weights.IMAGENET1K_V1},
}


class EfficientNet(NNEfficientNet):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        last_channel: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super(EfficientNet, self).__init__(inverted_residual_setting, dropout, stochastic_depth_prob, num_classes,
                                           norm_layer, last_channel, **kwargs)
        convert_model(self, **kwargs)

        # Set eps of BN 1e-3 for EfficientNetV2
        if FusedMBConvConfig in [type(setting) for setting in inverted_residual_setting]:
            for module in self.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eps = 1e-3

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(torch.flatten(x, 1))
        return x


class EfficientNetBackbone(EfficientNet):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        last_channel: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super(EfficientNetBackbone, self).__init__(inverted_residual_setting, dropout, stochastic_depth_prob,
                                                   num_classes, norm_layer, last_channel, **kwargs)
        delattr(self, 'avgpool')
        delattr(self, 'classifier')
        self.features = self.features[:-1]
        self.extraction_points = kwargs.pop('extraction_points')
        self.in_channels = [self.features[i][-1].blocks.proj_conv[0].out_channels for i in self.extraction_points]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        for i, feature in enumerate(self.features):
            x = feature(x)
            if i in self.extraction_points:
                features.append(x)
        return x, features
