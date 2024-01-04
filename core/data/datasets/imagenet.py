import os
import io
import h5py
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data


class ImageNet(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.num_classes = 1001
        self.transform = transform
        self.hdf5_root = os.path.join(root, 'train.hdf5') if train else os.path.join(root, 'val.hdf5')

        if not os.path.exists(self.hdf5_root):
            raise(RuntimeError("Cannot Find" + self.hdf5_root))

        self.length = len(h5py.File(self.hdf5_root, 'r')['binary_data'])
        if self.length == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

    def __getitem__(self, index):
        self.hdf5 = h5py.File(self.hdf5_root, 'r')
        img = Image.open(io.BytesIO(self.hdf5['binary_data'][index])).convert('RGB')  # Add convert for gray image
        img_array = np.asarray(img)[:, :, ::-1]  # convert RGB to BGR
        target = self.hdf5['labels'][index]
        if target is None:
            target = torch.zeros(1).long()

        sample = {'index': index, 'image': img_array, 'class': target, 'height': img.height, 'width': img.width}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.length
