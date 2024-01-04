import copy
import torch.nn.functional as F

from ..loss import Loss
from .multibox_loss import MultiBoxLoss
from ..classification import iou_aware_sigmoid_focal_loss

loss_list = ['focal', 'multi-box']


class DetectionLoss(Loss):
    def __init__(self, type, prior, iou_aware=False, loc_weight=1.0,
                 mix_up=False, label_smoothing=False):
        super(DetectionLoss, self).__init__()

        assert len(type) == 2, 'Detection needs classification loss & regression loss both'

        self.loss = MultiBoxLoss(type[0], type[1], 0.5, prior, iou_aware=iou_aware, loc_weight=loc_weight,
                                 mix_up=mix_up, label_smoothing=label_smoothing)
        self.cls_loss_type = type[0]
        self.loc_weight = loc_weight
        self.loss_dict = {'odm_loc': 0, 'odm_conf': 0}

    def loss_with_gt(self, preds, gt):
        loss = 0
        if 'detection' in gt:
            odm_loc, odm_conf = self.loss(preds['detection'], gt['detection'])
            self.loss_dict['odm_loc'] += odm_loc.item()
            self.loss_dict['odm_conf'] += odm_conf.item()
            loss += odm_loc + odm_conf

            if hasattr(self, 'arm_loss'):
                binary_targets = copy.deepcopy(gt['detection'])
                for target in binary_targets:
                    if target.nelement():
                        target[:, 4] = target[:, 4] > 0
                arm_loc, arm_conf = self.arm_loss(preds['detection'][0:2], binary_targets)
                self.loss_dict['arm_loc'] += arm_loc.item()
                self.loss_dict['arm_conf'] += arm_conf.item()
                loss += arm_loc + arm_conf
        return loss

    def loss_with_pseudo(self, preds, pseudo):
        loss = 0
        if 'detection' in pseudo:
            thr = 0.3
            if self.cls_loss_type == 'focal':
                odm_target = F.sigmoid(pseudo['detection'][-1])
                odm_conf = iou_aware_sigmoid_focal_loss(preds['detection'][-1], odm_target, 0.25, 2, thr=thr,
                                                        reduction='sum')
                odm_valid = odm_target.max(dim=-1)[0] > thr
                odm_conf = odm_conf / odm_valid.sum()
            else:
                odm_target = F.softmax(pseudo['detection'][-1], dim=-1)
                odm_log_prob = F.log_softmax(preds['detection'][-1], dim=-1)
                odm_valid = odm_target.max(dim=-1)[0] > thr
                odm_conf = -(odm_target * odm_log_prob)[odm_valid].sum() / odm_valid.sum()
            odm_loc = F.smooth_l1_loss(preds['detection'][-2][odm_valid], pseudo['detection'][-2][odm_valid])

            self.loss_dict['odm_loc'] += odm_loc.item()
            self.loss_dict['odm_conf'] += odm_conf.item()
            loss += self.loc_weight * odm_loc + odm_conf

            if hasattr(self, 'arm_loss'):
                arm_target = F.softmax(pseudo['detection'][1], dim=-1)
                arm_valid = arm_target.max(dim=-1)[0] > 0
                arm_log_prob = F.log_softmax(preds['detection'][1], dim=-1)
                arm_conf = -(arm_target * arm_log_prob)[arm_valid].sum() / arm_valid.sum()
                arm_loc = F.smooth_l1_loss(preds['detection'][0][arm_valid], pseudo['detection'][0][arm_valid])

                self.loss_dict['arm_loc'] += arm_loc.item()
                self.loss_dict['arm_conf'] += arm_conf.item()
                loss += arm_loc + arm_conf
        return loss
