import random
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .collate_fn import collate_fn
from .datasets import COCODetection, VOCDetection, NYUDepthEstimation, NYU_MT, ImageNet
from .augmentations import BaseTransform, RandScale

dataset_list = ['VOC', 'COCO', 'NYU', 'NYU_MT', 'ImageNet']


def build_train_dataset(datasets, year, applications, dataset_root, size, min_scale, max_scale):
    rgb_means = (0.485, 0.456, 0.406)
    transform = RandScale(size, min_scale, max_scale, rgb_means)

    train_sets = []
    for dataset in datasets:
        assert dataset in dataset_list, '%s is not supported' % dataset
        if dataset == 'VOC':
            assert year in ['2007', '2012'], 'Only 2007 and 2012 versions are supported for VOC dataset'
            if 'detection' in applications and 'segmentation' in applications:
                train_images = [('2007', 'trainvalsegdet'), ('2012', 'trainvalsegdet')]
            elif 'detection' in applications:
                # object detection for multi-task baseline
                train_images = [('2007', 'trainvalsegdet'), ('2012', 'trainvalsegdet')]

                # Typical VOC dataset config for object detection
                # train_images = [('2007', 'trainval'), ('2012', 'trainval')]
            elif 'segmentation' in applications:
                # semantic segmentation for multi-task baseline
                train_images = [('2007', 'trainvalsegdet'), ('2012', 'trainvalsegdet')]

                # Typical VOC dataset config for semantic segmentation
                # train_images = [('2012', 'trainaug')]
            else:  # Training without labels
                train_images = [('2007', 'trainvalsegdet'), ('2012', 'trainvalsegdet')]
            train_dataset = VOCDetection(dataset_root + '/VOC/', transform=transform, image_sets=train_images)
        elif dataset == 'COCO':
            train_dataset = COCODetection(dataset_root + '/mscoco/', 'train2017', transform=transform)
        elif dataset == 'NYU':
            train_dataset = NYUDepthEstimation(dataset_root + '/NYU/', train=True, transform=transform)
        elif dataset == 'NYU_MT':
            train_dataset = NYU_MT(dataset_root + '/NYUD_MT', train=True, transform=transform)
        elif dataset == 'ImageNet':
            train_dataset = ImageNet(dataset_root + '/ImageNet/', train=True, transform=transform)
        train_sets.append(train_dataset)
    return ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]


def build_test_datasets(datasets, year, applications, size, dataset_root):
    rgb_means = (0.485, 0.456, 0.406)
    transform = BaseTransform(size, rgb_means)

    test_sets = list()
    for dataset in datasets:
        test_set = []
        assert dataset in dataset_list, f'{dataset} is not supported'
        if dataset == 'VOC':
            assert year in ['2007', '2012'], 'Only 2007 and 2012 versions are supported for VOC dataset'
            assert 'detection' in applications or 'segmentation' in applications, \
                'VOC dataset only supports detection and segmentation'
            if 'detection' in applications and 'segmentation' in applications:
                test_images = [('2007', 'test'), ('2012', 'segval')]
            elif 'detection' in applications:
                test_images = [('2007', 'test'), ('2012', 'segval')]
                # test_images = [('2007', 'test')] if year == '2007' else [('2012', 'val')]  # typical VOC setup
            elif 'segmentation' in applications:
                test_images = [('2012', 'segval')]
                # test_images = [('2007', 'test'), ('2012', 'segval')]  # typical VOC setup
            test_set = [VOCDetection(dataset_root + '/VOC/', train=False, transform=transform, image_sets=test_images)]
        elif dataset == 'COCO':
            test_set = [COCODetection(dataset_root + '/mscoco/', 'val2017', transform=transform)]
        elif dataset == 'NYU':
            test_set = [NYUDepthEstimation(dataset_root + '/NYU/', train=False, transform=transform)]
        elif dataset == 'NYU_MT':
            test_set = [NYU_MT(dataset_root + '/NYUD_MT', train=False, transform=transform)]
        test_sets += test_set
    return test_sets


def build_data_loaders(args):
    train_set = build_train_dataset(args.train_dataset, args.year, args.app, args.dataset_root,
                                    args.training_size, args.min_scale, args.max_scale)
    train_sampler = DistributedSampler(train_set, shuffle=True) if dist.is_initialized() else None
    train_loader = DataLoader(train_set, args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers,
                              drop_last=True, sampler=train_sampler, pin_memory=True, collate_fn=collate_fn)

    test_sets = build_test_datasets(args.test_dataset, args.year, args.app, args.size, args.dataset_root)
    test_loaders = []
    for test_set in test_sets:
        test_sampler = DistributedSampler(test_set, shuffle=False) if dist.is_initialized() else None
        test_loaders.append(DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                       sampler=test_sampler, collate_fn=collate_fn, pin_memory=True))

    return train_loader, test_loaders


def get_batch(batch_iterator, loader, epoch, use_cuda=True):
    try:
        sample = next(batch_iterator)
    except StopIteration:
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)
        batch_iterator = iter(loader)
        sample = next(batch_iterator)

    device = 'cuda' if use_cuda else 'cpu'
    return batch_iterator, {key: [anno.to(device) for anno in value] if type(value) is list else value.to(device)
                            for key, value in sample.items()}


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.num_classes = self.datasets[0].num_classes

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            else:
                idx -= len(dataset)
        return idx

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __str__(self):
        return ' + '.join([str(dataset) for dataset in self.datasets])


class BalancedConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.num_classes = 21  # ToDo
        self.max_len = max([len(dataset) for dataset in datasets])
        self.length = self.max_len * len(datasets)
        self.excessive_idx = [[i for i in range(len(dataset))] for dataset in datasets]
        [random.shuffle(excessive_idx) for excessive_idx in self.excessive_idx]

    def __getitem__(self, idx):
        if idx == 1:
            [random.shuffle(excessive_idx) for excessive_idx in self.excessive_idx]

        for dataset, excessive_idx in zip(self.datasets, self.excessive_idx):
            if idx < self.max_len:
                return dataset[idx if idx < len(dataset) else excessive_idx[idx % len(dataset)]]
            else:
                idx -= self.max_len
        return idx

    def __len__(self):
        return self.length

    def __str__(self):
        return ' + '.join([str(dataset) for dataset in self.datasets])
