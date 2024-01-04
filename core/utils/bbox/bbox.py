import torch
from math import sqrt as sqrt
from itertools import product as product
from abc import ABCMeta, abstractmethod


def anchor_generator(cfg, levels):
    image_size = cfg['min_dim']
    # number of priors for feature map location (either 4 or 6)
    feature_maps = cfg['feature_maps'][:levels]
    min_sizes = cfg['min_sizes'][:levels]
    steps = cfg['steps'][:levels]
    aspect_ratios = cfg['aspect_ratios'][:levels]
    anchor_sizes = cfg['sizes'][:levels]
    bbox_type = cfg['bbox_type']
    clip = cfg['clip']

    if len(image_size) == 1:
        image_size = [image_size[0], image_size[0]]
        for i in range(len(feature_maps)):
            feature_maps[i] = [feature_maps[i], feature_maps[i]]

    prior_box = []
    for k, f in enumerate(feature_maps):
        f_k_x = image_size[0] / steps[k]
        f_k_y = image_size[1] / steps[k]
        s_k_x = min_sizes[k] / image_size[0]
        s_k_y = min_sizes[k] / image_size[1]

        # Default size (1)and aspect ratio (1)
        default = torch.stack([torch.Tensor([(j + 0.5) / f_k_x, (i + 0.5) / f_k_y, s_k_x, s_k_y])
                               for i, j in product(range(f[0]), range(f[1]))])

        cxcywh = None
        # Generate anchors with various sizes and aspect ratios
        for a_s in anchor_sizes[k]:
            # Generate anchors with anchor_size and aspect ratio of 1
            new_default = torch.cat([default[:, :2], default[:, 2:] * torch.tensor([a_s, a_s])], dim=1)
            cxcywh = new_default.unsqueeze(dim=0) if cxcywh is None \
                else torch.cat([cxcywh, new_default.unsqueeze(dim=0)])

            # Generate anchors with aspect_ratio and 1 / aspect ratio
            for ar in aspect_ratios[k]:
                ar1 = torch.cat([new_default[:, :2], new_default[:, 2:] * torch.tensor([sqrt(ar), 1 / sqrt(ar)])], dim=1)
                ar2 = torch.cat([new_default[:, :2], new_default[:, 2:] * torch.tensor([1 / sqrt(ar), sqrt(ar)])], dim=1)
                cxcywh = torch.cat([cxcywh, torch.stack([ar1, ar2])])
        prior_box += cxcywh.permute([1, 0, 2]).reshape(-1, 4)
    prior_box = torch.stack(prior_box)
    if clip:
        prior_box.clamp_(max=1, min=0)
    prior_box = prior_box.cuda() if torch.cuda.is_available() else prior_box

    return prior_box


def bbox_factory(prior_box, bbox_type='SSD', variance=(0.1, 0.2)):
    if bbox_type == 'SSD':
        return SSDBBox(prior_box, variance)
    elif bbox_type == 'YoLo':
        return YoLoBBox(prior_box, variance)
    elif bbox_type == 'simple':
        return SimpleBBox(prior_box, variance)


class BBox(metaclass=ABCMeta):
    def __init__(self, prior_box, variance):
        """
        :param prior_box: (tensor) Pre-defined boxes of which format is always 'cxcywh',
                                shape: [num_priors, 4]
        :param variance: (list[float]) Variances of bbox boxes
        """
        self.prior_box = prior_box  # Priors is always 'cxcywh' format
        self.variance = variance

    def get_prior_num(self):
        return self.prior_box.size(0)

    @abstractmethod
    def encode(self, gt):
        """
        Args:
            gt (tensor): ground truth bounding box of which format is 'xyxy',
                Shape: [num_priors,4]
        Return:
            encoded prediction target
        """
        pass

    @abstractmethod
    def decode(self, loc):
        """
        Decode locations from predictions using priors to undo
            the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
        Return:
            decoded bounding box predictions of which is 'xyxy'
        """
        pass


class SSDBBox(BBox):
    def __init__(self, prior_box, variance):
        super(SSDBBox, self).__init__(prior_box, variance)

    def encode(self, gt, offset=None):
        prior_box = offset.detach() if offset is not None else self.prior_box

        if len(prior_box.shape) != 3:
            prior_box = torch.unsqueeze(prior_box, dim=0)

        if len(gt.shape) != 3:
            gt = torch.unsqueeze(gt, dim=0)

        # dist b/t match center and prior's center
        g_cxcy = (gt[:, :, :2] + gt[:, :, 2:])/2 - prior_box[:, :, :2]
        # encode variance
        g_cxcy /= (self.variance[0] * prior_box[:, :, 2:])
        # match wh / prior wh
        g_wh = (gt[:, :, 2:] - gt[:, :, :2]) / prior_box[:, :, 2:]
        g_wh = torch.log(g_wh) / self.variance[1]

        return torch.cat([g_cxcy, g_wh], dim=-1)  # [batch, num_priors,4]

    # Adapted from https://github.com/Hakuyume/chainer-ssd
    def decode(self, loc, offset=None):
        prior_box = offset.detach() if offset is not None else self.prior_box
        if len(prior_box.shape) != 3:
            prior_box = torch.unsqueeze(prior_box, dim=0)

        if len(loc.shape) != 3:
            loc = torch.unsqueeze(loc, dim=0)

        boxes = torch.cat((prior_box[:, :, :2] + loc[:, :, :2] * self.variance[0] * prior_box[:, :, 2:],
                           prior_box[:, :, 2:] * torch.exp(loc[:, :, 2:] * self.variance[1])), dim=-1)
        boxes[:, :, :2] -= boxes[:, :, 2:] / 2
        boxes[:, :, 2:] += boxes[:, :, :2]
        return boxes


class YoLoBBox(BBox):
    def __init__(self, prior_box, variance):
        super(YoLoBBox, self).__init__(prior_box, variance)

        self.eliminate_grid_factor = 1.05

    def encode(self, gt, offset=None):
        pass

    def decode(self, loc, offset=None):
        prior_box = offset.detach() if offset is not None else self.prior_box
        if len(prior_box.shape) != 3:
            prior_box = torch.unsqueeze(prior_box, dim=0)

        if len(loc.shape) != 3:
            loc = torch.unsqueeze(loc, dim=0)

        boxes = torch.cat((torch.sigmoid(loc[:, :, :2]) * self.eliminate_grid_factor + prior_box[:, :, :2],
                           prior_box[:, :, 2:] * torch.exp(loc[:, :, 2:])), dim=-1)
        boxes[:, :, :2] -= boxes[:, :, 2:] / 2
        boxes[:, :, 2:] += boxes[:, :, :2]
        return boxes


class SimpleBBox(BBox):
    def __init__(self, prior_box, variance):
        super(SimpleBBox, self).__init__(prior_box, variance)

    def encode(self, gt, offset=None):
        prior_box = offset.detach() if offset is not None else self.prior_box

        if len(prior_box.shape) != 3:
            prior_box = torch.unsqueeze(prior_box, dim=0)

        if len(gt.shape) != 3:
            gt = torch.unsqueeze(gt, dim=0)

        g_cxcy = (gt[:, :, :2] + gt[:, :, 2:]) / 2 - prior_box[:, :, :2]
        g_wh = (gt[:, :, 2:] - gt[:, :, :2]) - prior_box[:, :, 2:]

        # return target
        return torch.cat([g_cxcy, g_wh], dim=-1)  # [batch,num_priors,4]

    def decode(self, loc, offset=None):
        prior_box = offset.detach() if offset is not None else self.prior_box
        if len(prior_box.shape) != 3:
            prior_box = torch.unsqueeze(prior_box, dim=0)

        if len(loc.shape) != 3:
            loc = torch.unsqueeze(loc, dim=0)

        boxes = prior_box + loc

        boxes[:, :, :2] -= boxes[:, :, 2:] / 2
        boxes[:, :, 2:] += boxes[:, :, :2]
        return boxes
