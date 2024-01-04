import numpy as np
from typing import Tuple
from abc import abstractmethod

import cv2
import torch
from numpy import random
from torch.nn import Module
from functools import partial
from torchvision.transforms import functional_tensor as F


class Geometric(Module):
    def transform(self, sample, *args, **kwargs):
        self._image(sample, *args, **kwargs)
        if 'target' in sample:
            self._detection(sample, *args, **kwargs)
        if 'label_map' in sample:
            self._segmentation(sample, *args, **kwargs)
        if 'depth' in sample:
            self._depth(sample, *args, **kwargs)
        if 'normal' in sample:
            self._surface_normal(sample, *args, **kwargs)
        return sample

    @abstractmethod
    def _image(self, sample, *args, **kwargs):
        pass

    @abstractmethod
    def _detection(self, sample, *args, **kwargs):
        pass

    @abstractmethod
    def _segmentation(self, sample, *args, **kwargs):
        pass

    @abstractmethod
    def _depth(self, sample, *args, **kwargs):
        pass

    @abstractmethod
    def _surface_normal(self, sample, *args, **kwargs):
        pass


class Expand(Geometric):
    def __init__(self, value, ignore_index, invalid):
        super(Expand, self).__init__()
        self._value = value
        self._ignore_index = ignore_index
        self._invalid = invalid

    def forward(self, sample, size: Tuple[int, int], top: int, left: int, scale: float):
        height, width = sample['image'].shape[:2]
        copy = partial(cv2.copyMakeBorder, top=top, bottom=size[0] - height - top,
                       left=left, right=size[1] - width - left, borderType=cv2.BORDER_CONSTANT)
        return self.transform(sample, copy, top, left, scale)

    def _image(self, sample, copy=None, *args, **kwargs):
        sample['image'] = copy(sample['image'], value=self._value)

    def _detection(self, sample, copy=None, top=None, left=None, *args, **kwargs):
        sample['target'][:, :4] += (left, top, left, top)

    def _segmentation(self, sample, copy=None, *args, **kwargs):
        sample['label_map'] = copy(sample['label_map'], value=self._ignore_index)

    def _depth(self, sample, copy=None, top=None, left=None, scale=1.0, *args, **kwargs):
        sample['depth'] = copy(sample['depth'], value=self._invalid) * scale

    def _surface_normal(self, sample, copy=None, *args, **kwargs):
        sample['normal'] = copy(sample['normal'], value=self._invalid)


class Crop(Geometric):
    def forward(self, sample: dict, size: Tuple[int, int], top: int, left: int, scale: float):
        height, width, _ = sample['image'].shape
        bottom, right = int(top + size[0]), int(left + size[1])

        box = (top, bottom, left, right)
        self.transform(sample, box, scale)

    def _image(self, sample, bbox=None, *args, **kwargs):
        top, bottom, left, right = bbox
        sample['image'] = sample['image'][top:bottom, left:right]

    def _detection(self, sample, bbox=None, *args, **kwargs):
        top, bottom, left, right = bbox
        boxes = sample['target'][:, :4].copy()

        # adjust boundary of boxes
        boxes = np.maximum(boxes, (left, top, left, top))
        boxes = np.minimum(boxes, (right, bottom, right, bottom))
        mask = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) != 0

        if not any(mask):
            sample.pop('target')
        else:
            # adjust to crop (translation by subtracting crop's left,top)
            boxes = boxes[mask] - (left, top, left, top)
            labels = sample['target'][mask, -1].copy()
            sample['target'] = np.concatenate((boxes, np.reshape(labels, (-1, 1))), axis=1)

    def _segmentation(self, sample, bbox=None, *args, **kwargs):
        top, bottom, left, right = bbox
        sample['label_map'] = sample['label_map'][top:bottom, left:right]

    def _depth(self, sample, bbox=None, scale=None, *args, **kwargs):
        top, bottom, left, right = bbox
        sample['depth'] = sample['depth'][top:bottom, left:right] * scale

    def _surface_normal(self, sample, bbox=None, *args, **kwargs):
        top, bottom, left, right = bbox
        sample['normal'] = sample['normal'][top:bottom, left:right]


class RandomHorizontalFlip(Geometric):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p

    def forward(self, sample):
        if torch.rand(1).item() < self.p:
            self.transform(sample)
        return sample

    def _image(self, sample, *args, **kwargs):
        sample['image'] = F.hflip(sample['image'])

    def _detection(self, sample, *args, **kwargs):
        assert (sample['target'][:, :4] > 1).sum() == 0, "Horizontal flip must operate using percent coordinates."
        sample['target'][:, (0, 2)] = 1 - sample['target'][:, (2, 0)]

    def _segmentation(self, sample, *args, **kwargs):
        sample['label_map'] = F.hflip(sample['label_map'])

    def _depth(self, sample, *args, **kwargs):
        sample['depth'] = F.hflip(sample['depth'])

    def _surface_normal(self, sample, *args, **kwargs):
        sample['normal'] = F.hflip(sample['normal'])
        sample['normal'][0] = -sample['normal'][0]


class Resize(Geometric):
    def __init__(self, size, mean, ignore_index=255, invalid=0):
        super().__init__()
        self.mean = tuple([m * 255 for m in mean])
        self.size = size if isinstance(size, (list, tuple)) else (size, size)  # (h, w)
        self.ignore_index = ignore_index
        self.invalid = invalid
        self._expand = Expand(value=self.mean, ignore_index=self.ignore_index, invalid=self.invalid)

    def __call__(self, sample):
        h, w, _ = sample['image'].shape
        aspect_aware_size = self._get_aspect_aware_size(h, w)
        expanded_size = (int(aspect_aware_size[0] + 0.5), int(aspect_aware_size[1] + 0.5))
        self._expand(sample, expanded_size, 0, 0, 1.0)

        _h, _w, _ = sample['image'].shape
        if (_h, _w) == self.size:
            return sample
        return self.transform(sample, self.size[::-1], _h, _w)

    def _get_aspect_aware_size(self, h, w) -> Tuple[float, float]:
        target_ratio, sample_ratio = self.size[1] / self.size[0], w / h
        if sample_ratio == target_ratio:
            return h, w
        elif sample_ratio > target_ratio:
            return w * target_ratio, w
        else:
            return h, h * target_ratio

    def _image(self, sample, aspect_aware_size=None, *args, **kwargs):
        sample['image'] = cv2.resize(sample['image'], aspect_aware_size, interpolation=cv2.INTER_AREA)

    def _detection(self, sample, aspect_aware_size=None, _h=None, _w=None, *args, **kwargs):
        sample["target"][:, [0, 2]] *= aspect_aware_size[0] / _w
        sample["target"][:, [1, 3]] *= aspect_aware_size[1] / _h

    def _segmentation(self, sample, aspect_aware_size=None, *args, **kwargs):
        sample['label_map'] = cv2.resize(sample['label_map'], aspect_aware_size, interpolation=cv2.INTER_NEAREST)

    def _depth(self, sample, aspect_aware_size=None, *args, **kwargs):
        sample['depth'] = cv2.resize(sample['depth'], aspect_aware_size, interpolation=cv2.INTER_NEAREST)

    def _surface_normal(self, sample, aspect_aware_size=None, *args, **kwargs):
        sample['normal'] = cv2.resize(sample['normal'], aspect_aware_size, interpolation=cv2.INTER_AREA)


class RandomScale(Resize):
    def __init__(self, size, min_scale, max_scale, mean, ignore_index=255, invalid=0):
        super(RandomScale, self).__init__(size, mean, ignore_index, invalid)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._crop = Crop()

    def __call__(self, sample):
        # aspect ratio aware resize
        h, w, _ = sample['image'].shape
        aspect_aware_size = self._get_aspect_aware_size(h, w)
        expanded_size = (int(aspect_aware_size[0] + 0.5), int(aspect_aware_size[1] + 0.5))
        self.expand(sample, expanded_size, 1.0)

        if random.randint(2):
            self._rand_scale(sample, aspect_aware_size)

        _h, _w, _ = sample['image'].shape
        return self.transform(sample, self.size[::-1], _h, _w)

    def _rand_scale(self, sample, aspect_aware_size):
        scale = random.uniform(self.min_scale, self.max_scale)
        scaled_size = (int(aspect_aware_size[0] * scale + 0.5), int(aspect_aware_size[1] * scale + 0.5))
        if scale > 1.0:
            self.expand(sample, scaled_size, scale)
        else:
            self.random_sample(sample, scaled_size, scale)

    def expand(self, sample: dict, size: Tuple[int, int], scale):
        height, width, depth = sample['image'].shape
        assert size[0] - height >= 0, f"Expanded height ({size[0]}) is smaller than given ({height})"
        assert size[1] - width >= 0,  f"Expanded width ({size[1]}) is smaller than given ({width})"

        top = random.randint(0, size[0] - height + 1)  # [low, high)
        left = random.randint(0, size[1] - width + 1)
        self._expand(sample, size, top, left, scale)

    def random_sample(self, sample: dict, size: Tuple[int, int], scale: float):
        height, width, _ = sample['image'].shape
        assert size[0] - height <= 0, f"Cropped height ({size[0]}) is larger than given ({height})"
        assert size[1] - width <= 0,  f"Cropped width ({size[1]}) is larger than given ({width})"

        top = random.randint(0, height - size[0] + 1)
        left = random.randint(0, width - size[1] + 1)
        self._crop(sample, size, top, left, scale)
