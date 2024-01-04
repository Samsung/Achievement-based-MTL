from typing import Tuple, List
import torch
from .mobilenet_v2 import MobileNetV2
from .resnet import ResNetBackbone
from .efficientnet import EfficientNetBackbone


def backbone_factory(name: str, **kwargs) -> Tuple[torch.nn.Module, List[int]]:
    state_dict = None
    if 'resnet' in name:
        from .resnet import model_config_dict, get_state_dict
        assert name in model_config_dict, f"'{name} if not supported."
        model_config = model_config_dict[name]

        block, layers = model_config['arg_dict']
        replace_stride_with_dilation = kwargs.pop('replace_stride_with_dilation', [None, True, True])
        args_dict = {'num_classes': 1000}
        args_dict.update(**kwargs)
        args_dict['replace_stride_with_dilation'] = replace_stride_with_dilation
        model = ResNetBackbone(block, layers, **args_dict)
        state_dict = get_state_dict(name)
    elif 'mobilenet' in name:
        from .mobilenet_v2 import model_config_dict
        model_config = model_config_dict[name]
        model = MobileNetV2(**kwargs)
        state_dict = model_config['weights'].get_state_dict(progress=True)
    elif 'efficientnet' in name:
        from .efficientnet import model_config_dict
        model_config = model_config_dict[name]
        activation = kwargs.pop('activation', torch.nn.SiLU)
        assert name in model_config_dict, NotImplementedError(name + 'is not supported')

        # default values
        args_dict = {'num_classes': 1000, 'se_ratio': 0.25, 'activation': activation, 'se_activation': torch.nn.SiLU}
        if 'skimmed' in kwargs:
            args_dict['activation'] = 'relu6'
            args_dict['se_ratio'] = None
        args_dict['extraction_points'] = model_config['extraction_points']
        inverted_residual_setting, last_channel = model_config['arg_dict']
        model = EfficientNetBackbone(inverted_residual_setting, 0.2, last_channel=last_channel, **args_dict)
        state_dict = None if kwargs['from_scratch'] else model_config['weights'].get_state_dict(True)

    if state_dict:
        new_state_dict = {}
        for (key, _value), (_key, value) in zip(model.state_dict().items(), state_dict.items()):
            new_state_dict[key] = value
        model.load_state_dict(new_state_dict, strict=False)
    return model, model.in_channels
