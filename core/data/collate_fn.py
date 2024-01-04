import torch


def collate_fn(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """

    sample = dict()
    sample['images'] = torch.stack([value['image'] for value in batch], 0)
    sample['heights'] = [value['height'] for value in batch]
    sample['widths'] = [value['width'] for value in batch]
    sample['index'] = [value['index'] for value in batch]

    _, h, w = batch[0]['image'].shape
    if sum(['target' in sample for sample in batch]):
        sample['detection'] = [value['target'] if 'target' in value else torch.tensor([], device='cpu').float()
                               for value in batch]
    if sum(['label_map' in sample for sample in batch]):
        sample['segmentation'] = torch.stack([value['label_map'] if 'label_map' in value else
                                              torch.ones([h, w], dtype=torch.uint8, device='cpu').fill_(255)
                                              for value in batch], 0)
    if sum(['depth' in sample for sample in batch]):
        sample['depth'] = torch.stack([value['depth'] if 'depth' in value else
                                       torch.zeros([1, h, w], dtype=torch.int16, device='cpu')
                                       for value in batch], 0)
    if sum(['normal' in sample for sample in batch]):
        sample['normal'] = torch.stack([value['normal'] if 'normal' in value else
                                       torch.zeros([3, h, w], dtype=torch.float32, device='cpu')
                                       for value in batch], 0)
    return sample
