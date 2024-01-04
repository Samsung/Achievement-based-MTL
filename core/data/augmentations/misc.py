import torch
import numpy as np


class ToAbsoluteCoord(object):
    def __call__(self, sample):
        height, width, channels = sample['image'].shape
        if 'target' in sample:
            sample['target'][:, :4] *= (width, height, width, height)
        return sample


class ToPercentCoord(object):
    def __call__(self, sample):
        height, width, channels = sample['image'].shape
        if 'target' in sample:
            sample['target'][:, :4] /= (width, height, width, height)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample['image'] = torch.from_numpy(sample['image']).permute(2, 0, 1).contiguous()
        sample['height'] = torch.tensor(sample['height'], dtype=torch.int32, device='cpu')
        sample['width'] = torch.tensor(sample['width'], dtype=torch.int32, device='cpu')

        if 'index' in sample:
            sample['index'] = torch.tensor(sample['index'], device='cpu')
        if 'target' in sample:
            sample['target'] = torch.from_numpy(sample['target']).float()
        if 'label_map' in sample:
            sample['label_map'] = torch.from_numpy(sample['label_map'])
        if 'depth' in sample:
            sample['depth'] = torch.from_numpy(np.int16(sample['depth'])).unsqueeze(0)
        if 'normal' in sample:
            sample['normal'] = torch.from_numpy(sample['normal']).permute(2, 0, 1).contiguous()
        return sample
