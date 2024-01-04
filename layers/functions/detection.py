import torch
import torch.nn.functional as F
import torchvision.ops as ops


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, variance, top_k, conf_thresh, nms_thresh, objectness_thr, keep_top_k):
        self.num_classes = num_classes
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.objectness_thr = objectness_thr
        self.variance = variance

    def detect(self, preds, prior):
        """
        Args:
            loc: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        assert len(preds) == 2 or len(preds) == 4
        if len(preds) == 2:
            loc, conf = preds
            arm_loc, arm_conf = None, None
        else:
            arm_loc, arm_conf, loc, conf = preds

        sigmoid_mode = (conf.size(-1) != self.num_classes)
        if arm_conf is not None:
            arm_conf = F.sigmoid(arm_conf) if sigmoid_mode else F.softmax(arm_conf, dim=-1)
        conf = F.sigmoid(conf) if sigmoid_mode else F.softmax(conf, dim=-1)

        # non-negative filtering
        if arm_conf is not None:
            arm_object_conf = arm_conf.data[:, :, 1:]
            no_object_index = arm_object_conf <= self.objectness_thr
            conf[no_object_index.expand_as(conf)] = 0

        num = loc.size(0)  # batch size
        num_priors = prior.get_prior_num()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        output = torch.zeros([num, self.num_classes, self.top_k, 5], device=device)
        conf_preds = conf.view(num, num_priors, self.num_classes - sigmoid_mode).transpose(2, 1)

        decoded_boxes = prior.decode(loc, arm_loc)

        conf_mask = conf_preds.gt(self.conf_thresh)
        for i in range(num):
            mask = conf_mask[i]
            decoded_box = decoded_boxes[i]
            conf_score = conf_preds[i]
            for cl in range(1, self.num_classes):
                box = decoded_box[mask[cl - sigmoid_mode]]
                score = conf_score[cl - sigmoid_mode][mask[cl - sigmoid_mode]]
                if score.shape[0] > self.top_k * 10:
                    idx = score.sort(descending=True)[1][:self.top_k * 10]
                    box = box[idx]
                    score = score[idx]
                keep_index = ops.nms(box, score, self.nms_thresh)[:self.top_k]
                output[i, cl, :len(keep_index), :] = torch.cat(
                    [box[keep_index], torch.unsqueeze(score[keep_index], dim=-1)], dim=-1)

        flt = output.view(num, -1, 5)
        _, idx = flt[:, :, -1].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank >= self.keep_top_k).unsqueeze(-1).expand_as(flt)] = 0
        return output
