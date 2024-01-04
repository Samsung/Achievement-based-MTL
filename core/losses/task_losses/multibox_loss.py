import torch
import torch.nn as nn
from torchvision import ops

from .box_utils import match, ATSSmatch, log_sum_exp
from ..classification import CrossEntropyLoss, FocalLoss
from ..regression.iou_loss import IoULoss


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cls_loss_type, loc_loss_type, overlap_thresh, priors, neg_pos=3, atss=True, objectness_thr=0,
                 iou_aware=False, loc_weight=1.0, mix_up=False, label_smoothing=False):
        super(MultiBoxLoss, self).__init__()
        self.threshold = overlap_thresh
        self.priors = priors
        self.do_neg_mining = (cls_loss_type != 'focal')
        self.neg_pos_ratio = neg_pos
        self.variance = [0.1, 0.2]
        self.objectness_thr = objectness_thr
        self.softmax = nn.Softmax(dim=2)
        self.mix_up = mix_up
        self.label_smoothing = label_smoothing
        self.num_priors = self.priors.get_prior_num()

        self.iou_aware = iou_aware
        self.loc_weight = loc_weight
        self.match = ATSSmatch if atss else match
        self.cls_loss_type = cls_loss_type
        self.loc_loss_type = loc_loss_type

        if self.cls_loss_type == 'CE':
            self.cls_loss = CrossEntropyLoss(reduction='sum')
        elif self.cls_loss_type == 'focal':
            self.cls_loss = FocalLoss(gamma=2, alpha=0.25, reduction='sum')
        else:
            raise AssertionError('%s is not yet supported.\n'
                                 'Only cross entropy (CE) and focal loss (focal) are supported now.' % cls_loss_type)

        if self.loc_loss_type == 'L1':
            self.reg_loss = nn.SmoothL1Loss(reduction='sum')
        elif self.loc_loss_type == 'IoU':
            self.reg_loss = IoULoss(reduction='sum')
            if 'YoLo' in str(type(self.priors)):
                raise AssertionError('%s is not yet supported with yolo style bounding box' % loc_loss_type)
        else:
            raise AssertionError('%s is not yet supported.\n'
                                 'Only L1 norm (L1) and IoU loss (IoU) are supported now.' % loc_loss_type)

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds and conf preds.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        if len(predictions) == 4:
            offsets, objectness, loc_data, conf_data = predictions
            offsets = torch.stack([data for i, data in enumerate(offsets) if targets[i].nelement()], 0)
            objectness = torch.stack([data for i, data in enumerate(objectness) if targets[i].nelement()], 0)
        elif len(predictions) == 2:
            loc_data, conf_data = predictions
            objectness, offsets = None, None
        else:
            raise NotImplementedError

        # Remove predictions from images with no labels
        loc_data = torch.stack([data for i, data in enumerate(loc_data) if targets[i].nelement()], 0)
        conf_data = torch.stack([data for i, data in enumerate(conf_data) if targets[i].nelement()], 0)
        targets = [target for target in targets if target.nelement()]

        num = loc_data.size(0)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, self.num_priors, 4)
        conf_t = torch.LongTensor(num, self.num_priors)
        if self.mix_up:
            mixup_ratio_t = torch.Tensor(num, self.num_priors)

        for idx in range(num):
            if self.mix_up:
                truths = targets[idx][:, :-2].detach()
                labels = targets[idx][:, -2].detach()
                mixups = targets[idx][:, -1].detach()
                self.match(self.threshold, truths, self.priors, self.variance, labels, loc_t, conf_t, idx, offsets,
                           mixups=mixups, mixup_ratio_t=mixup_ratio_t)
            else:
                truths = targets[idx][:, :-1].detach()
                labels = targets[idx][:, -1].detach()
                self.match(self.threshold, truths, self.priors, self.variance, labels, loc_t, conf_t, idx, offsets)
        if torch.cuda.is_available():
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            if self.mix_up:
                mixup_ratio_t = mixup_ratio_t.cuda()

        # Negative anchor filtering
        pos = conf_t > 0
        if objectness is not None:
            arm_conf = self.softmax(objectness.detach())[:, :, 1]
            objectless_idx = arm_conf <= self.objectness_thr
            pos[objectless_idx] = 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        if torch.sum(pos_idx) == 0:
            loss_l = (0 * loc_data).sum()
            loss_c = (0 * conf_data).sum()
            return loss_l, loss_c

        # prior = self.priors.repeat(num, 1, 1)[pos_idx].view(-1, 4)
        xyxy1 = self.priors.decode(loc_data.detach(), offsets)[pos_idx].view(-1, 4)
        xyxy2 = loc_t.detach()[pos_idx].view(-1, 4)
        iou = ops.box_iou(xyxy1, xyxy2).diagonal()

        if self.loc_loss_type == 'L1':  # decode bbox for iou loss
            loc_t = self.priors.encode(loc_t, offsets)

        if self.loc_loss_type == 'IoU':  # decode bbox for iou loss
            loc_data = self.priors.decode(loc_data, offsets)

        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        loss_l = self.reg_loss(loc_p, loc_t) * self.loc_weight

        num_classes = conf_data.size(-1)
        if self.do_neg_mining:
            # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, num_classes)
            loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

            # Hard Negative Mining
            loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
            loss_c = loss_c.view(num, -1)
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.sum(1, keepdim=True)
            num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=pos.size(1)-1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            # Confidence Loss Including Positive and Negative Examples
            conf_p = conf_data[pos].view(-1, num_classes)
            target_p = conf_t[pos]

            conf_n = conf_data[neg].view(-1, num_classes)
            target_n = conf_t[neg]
        else:
            conf_p = conf_data[pos].view(-1, num_classes)
            target_p = conf_t[pos]

            neg = torch.bitwise_not(pos)
            conf_n = conf_data[neg].view(-1, num_classes)
            target_n = conf_t[neg]

        loss_c_pos = self.cls_loss(conf_p, target_p, iou if self.iou_aware else None)
        loss_c_neg = self.cls_loss(conf_n, target_n)
        loss_c = loss_c_pos + loss_c_neg

        num_pos = pos.sum(1, keepdim=True)
        N = num_pos.detach().sum()
        loss_l /= N.float()
        loss_c /= N.float()

        return loss_l, loss_c
