from typing import Tuple

import torch
from torchvision import transforms

from .photometric import PhotometricDistort
from .geometric import Resize, RandomScale, RandomHorizontalFlip
from .misc import ToAbsoluteCoord, ToPercentCoord, ToTensor


class TensorTransform(torch.nn.Module):
    def __init__(self):
        super(TensorTransform, self).__init__()
        self.normalize = torch.jit.script(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )

    def forward(self, sample):
        sample = self.convert_from_ints(sample)
        sample['images'] = self.normalize(sample['images'])
        return sample

    @staticmethod
    def convert_from_ints(sample):
        sample['images'] = torch.divide(sample['images'].float(), 255.0)
        if 'segmentation' in sample:
            sample['segmentation'] = sample['segmentation'].long()
        if 'depth' in sample:
            sample['depth'] = torch.divide(sample['depth'].float(), 1000.0)
        return sample


class TensorDistortion(TensorTransform):
    def __init__(self):
        super().__init__()
        self.photometric_distortion = torch.nn.Sequential(
            PhotometricDistort(),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )

    def forward(self, sample):
        sample = self.convert_from_ints(sample)
        sample['images'] = self.photometric_distortion(sample['images'])
        return sample


class BaseTransform(torch.nn.Module):
    def __init__(self, size: Tuple[int, int], mean: Tuple[float, float, float]):
        super().__init__()
        self.augment = transforms.Compose([
            ToAbsoluteCoord(),
            Resize(size, mean),
            ToPercentCoord(),
            ToTensor(),
        ])

    def forward(self, sample):
        return self.augment(sample)


class RandScale(torch.nn.Module):
    """
    Applying geometric augmentation for the given image and ground-truth

    :param size: target size to be transformed
    :param mean: rgb mean of image
    :param ignore_index: ignore index for segmentation
    :param invalid: depth to be ignored
    """

    def __init__(self,
                 size: Tuple[int, int],
                 min_scale: float,
                 max_scale: float,
                 mean: Tuple[float, float, float],
                 ignore_index: int = 255,
                 invalid: int = 0):
        super().__init__()
        self.augment = transforms.Compose([
            ToAbsoluteCoord(),
            RandomScale(size, min_scale, max_scale, mean, ignore_index=ignore_index, invalid=invalid),
            ToPercentCoord(),
            ToTensor(),
            RandomHorizontalFlip(),
        ])

    def forward(self, sample):
        return self.augment(sample)
