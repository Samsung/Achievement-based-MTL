import torch
import math
from torchvision import ops


class IoULoss:
    """
    IoU Loss for resgression problem for bounding box
    Which maximizes Iou (Intersection On Uinon) between ground truth box and predicted box
    """

    def __init__(self, reduction=None):
        super(IoULoss, self).__init__()
        self.reduction = reduction

    def __call__(self, xyxy1, xyxy2):
        """
        Compute Complete IoU Loss
        :param xyxy1: predicted bboxes of which format is xyxy and shape is [N x 4]
        :param xyxy2: ground-truth bboxes of which format is xyxy and shape is [N x 4]
        :return: Complete IoU loss of the given bboxes
        """
        if xyxy1.shape[0] == 0:
            return xyxy1.sum() * 0

        iou = ops.box_iou(xyxy1, xyxy2).diagonal()
        c_l, c_r = torch.min(xyxy1[:, 0], xyxy2[:, 0]), torch.max(xyxy1[:, 2], xyxy2[:, 2])
        c_t, c_b = torch.min(xyxy1[:, 1], xyxy2[:, 1]), torch.max(xyxy1[:, 3], xyxy2[:, 3])

        cxcywh1 = ops.box_convert(xyxy1, 'xyxy', 'cxcywh')
        cxcywh2 = ops.box_convert(xyxy2, 'xyxy', 'cxcywh')
        inter_diag = (cxcywh1[:, :2] - cxcywh2[:, :2]).pow(2).sum(dim=-1)
        c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + torch.clamp((c_b - c_t), min=0) ** 2
        u = inter_diag / c_diag

        _, _, w1, h1 = cxcywh1.unbind(dim=-1)
        _, _, w2, h2 = cxcywh2.unbind(dim=-1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        with torch.no_grad():
            S = (iou > 0.5).float()
            alpha = S * v / (1 - iou + v)
        cious = iou - u - alpha * v
        cious = torch.clamp(cious, min=-1.0, max=1.0)

        loss = 1 - cious
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss
