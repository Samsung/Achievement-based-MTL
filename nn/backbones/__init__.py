from torch import nn
from .mobilenet_v2 import MobileNetV2
from .efficientnet import EfficientNet
from .resnet import ResNet


def get_backbone(name, progress=True, **kwargs) -> (nn.Module, nn.Module):
    state_dict, model_config_dict = None, None
    if name == 'mobilenet_v2':
        from .mobilenet_v2 import model_config_dict
        model_config = model_config_dict[name]
        model = MobileNetV2(**kwargs)
        state_dict = model_config['weights'].get_state_dict(progress=progress)
    elif 'resnet' in name:
        from .resnet import model_config_dict, get_state_dict
        assert name in model_config_dict, f"'{name} if not supported."
        model_config = model_config_dict[name]

        block, layers = model_config['arg_dict']
        args_dict = {'num_classes': 1000}
        args_dict.update(**kwargs)

        model = ResNet(block, layers, **args_dict)
        state_dict = get_state_dict(name)
    elif 'efficientnet' in name:
        from .efficientnet import model_config_dict
        model_config = model_config_dict[name]
        assert name in model_config_dict, NotImplementedError(name + 'is not supported')

        # default values
        args_dict = {'num_classes': 1000, 'se_ratio': 0.25, 'activation': nn.SiLU, 'se_activation': nn.SiLU}
        if 'skimmed' in kwargs:
            args_dict['activation'] = 'relu6'
            args_dict['se_ratio'] = None
        inverted_residual_setting, last_channel = model_config['arg_dict']
        model = EfficientNet(inverted_residual_setting, 0.2, last_channel=last_channel, **args_dict)
        state_dict = model_config['weights'].get_state_dict(progress=progress)
    transform = model_config['weights'].transforms()

    if state_dict:
        new_state_dict = {}
        for (key, _value), (_key, value) in zip(model.state_dict().items(), state_dict.items()):
            new_state_dict[key] = value
        model.load_state_dict(new_state_dict, strict=False)
    return model, transform
