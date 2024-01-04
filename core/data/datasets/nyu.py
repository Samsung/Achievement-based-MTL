import cv2
import numpy as np
from torch.utils.data import Dataset


class NYUDepthEstimation(Dataset):
    def __init__(self, root, train=True, transform=None):
        data_files = root + '/nyudepthv2_%s_files_with_gt.txt' % ('train' if train else 'test')
        with open(data_files, 'r') as f:
            self.nyu_dataset = f.readlines()
        self.path = root + '/%s/%s' % ('sync' if train else 'official_splits/test', '%s')
        self.transform = transform
        self.name = 'NYU v2'
        self.train = train
        self.num_classes = 27
        self.min_depth, self.max_depth = 0.001, 10

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx].split()
        image = cv2.imread(self.path % sample[0], cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(self.path % sample[1], cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        sample = {'image': image, 'depth': depth, 'height': height, 'width': width, 'index': idx}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

    def __str__(self):
        return self.name


class NYU_MT(Dataset):
    """
    We got NYU v2 multi_task dataset from https://github.com/facebookresearch/astmt
    """
    def __init__(self, root, train=True, transform=None):
        index_file = root + '/gt_sets/%s.txt' % ('train' if train else 'val')
        with open(index_file, 'r') as f:
            self.index = f.readlines()
        self.root = root
        self.name = 'NYU_v2'
        self.train = train
        self.num_classes = 40
        self.min_depth, self.max_depth = 0.001, 10
        self.transform = transform

    def __getitem__(self, idx):
        index = self.index[idx].replace('\n', '')
        image = cv2.imread(self.root + '/images/%s.jpg' % index, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(self.root + '/segmentation/%s.png' % index, cv2.IMREAD_UNCHANGED) - 1
        depth = np.load(self.root + '/depth/%s.npy' % index) * 1000
        normal = np.load(self.root + '/normals/%s.npy' % index)

        height, width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample = {'index': idx, 'image': image, 'depth': depth, 'label_map': label, 'normal': normal,
                  'height': height, 'width': width}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.index)

    def __str__(self):
        name = self.name
        transform = str(self.transform) if self.transform else ''
        return name + transform
