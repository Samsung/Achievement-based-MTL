import torch.nn as nn
from torch import Tensor

from typing import Callable, Optional, List
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.mobilenetv2 import MobileNetV2 as nnMobileNetV2
from .utils import convert_model

model_config_dict = {
    'mobilenet_v2': {'weights': MobileNet_V2_Weights.IMAGENET1K_V2},
}


class MobileNetV2(nnMobileNetV2):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(MobileNetV2, self).__init__(num_classes, width_mult, inverted_residual_setting, round_nearest, block,
                                          norm_layer)
        convert_model(self)
        self.extraction_points = [6, 13, 18]
        self.in_channels = [32, 96, 1280]
        delattr(self, 'classifier')

    def forward(self, x: Tensor) -> (Tensor, List[Tensor]):
        features = list()
        for i, feature in enumerate(self.features):
            x = feature(x)
            if i in self.extraction_points:
                features.append(x)
        return x, features
