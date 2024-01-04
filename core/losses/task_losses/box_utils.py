import torch
from torchvision.ops import box_iou


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]], 1)  # w, h


def ATSSmatch(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, offset=None, mixups=None, mixup_ratio_t=None):

    INF = 100000000

    # get prior_boxes
    if offset is not None:
        prior_boxes = priors.decode(offset[idx].detach()).squeeze(dim=0)
        offset[idx] = center_size(prior_boxes)
    else:
        prior_boxes = point_form(priors.prior_box)

    # TODO
    cfg = {
        6375: {
            'num_anchors_per_loc': 3,
            'num_anchors_per_level': [40 * 40 * 3, 20 * 20 * 3, 10 * 10 * 3, 5 * 5 * 3]},
        16320: {
            'num_anchors_per_loc': 3,
            'num_anchors_per_level': [64 * 64 * 3, 32 * 32 * 3, 16 * 16 * 3, 8 * 8 * 3]},
        25200: {
            'num_anchors_per_loc': 3,
            'num_anchors_per_level': [80 * 80 * 3, 40 * 40 * 3, 20 * 20 * 3]},
        25500: {
            'num_anchors_per_loc': 3,
            'num_anchors_per_level': [80 * 80 * 3, 40 * 40 * 3, 20 * 20 * 3, 10 * 10 * 3]},
        25575: {
            'num_anchors_per_loc': 3,
            'num_anchors_per_level': [80 * 80 * 3, 40 * 40 * 3, 20 * 20 * 3, 10 * 10 * 3, 5 * 5 * 3]},
    }

    bboxes_per_im = truths
    labels_per_im = labels
    anchors_per_im = prior_boxes
    num_gt = bboxes_per_im.shape[0]

    num_anchors_per_loc = cfg[priors.get_prior_num()]['num_anchors_per_loc']
    num_anchors_per_level = cfg[priors.get_prior_num()]['num_anchors_per_level']

    ious = box_iou(anchors_per_im, bboxes_per_im)

    gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
    gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
    gt_points = torch.stack((gt_cx, gt_cy), dim=1)

    anchors_cx_per_im = (anchors_per_im[:, 2] + anchors_per_im[:, 0]) / 2.0
    anchors_cy_per_im = (anchors_per_im[:, 3] + anchors_per_im[:, 1]) / 2.0
    anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

    distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

    # Selecting candidates based on the center distance between anchor box and object
    candidate_idxs = []
    star_idx = 0
    for level in range(len(num_anchors_per_level)):
        end_idx = star_idx + num_anchors_per_level[level]
        distances_per_level = distances[star_idx:end_idx, :]
        topk = min(9 * num_anchors_per_loc, num_anchors_per_level[level])
        _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
        candidate_idxs.append(topk_idxs_per_level + star_idx)
        star_idx = end_idx
    candidate_idxs = torch.cat(candidate_idxs, dim=0)

    # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
    candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
    iou_mean_per_gt = candidate_ious.mean(0)
    iou_std_per_gt = candidate_ious.std(0)
    iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
    is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

    # Limiting the final positive samples center to object
    anchor_num = anchors_cx_per_im.shape[0]
    for ng in range(num_gt):
        candidate_idxs[:, ng] += ng * anchor_num
    e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
    e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
    candidate_idxs = candidate_idxs.view(-1)
    l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
    t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
    r = bboxes_per_im[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
    b = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
    is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
    is_pos = is_pos & is_in_gts

    # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
    ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
    index = candidate_idxs.view(-1)[is_pos.view(-1)]
    ious_inf[index] = ious.t().contiguous().view(-1)[index]
    ious_inf = ious_inf.view(num_gt, -1).t()

    anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
    cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
    cls_labels_per_im[anchors_to_gt_values == -INF] = 0
    matched_gts = bboxes_per_im[anchors_to_gt_indexs]

    conf_t[idx] = cls_labels_per_im  # [num_priors] top class label for each prior
    if mixups is not None:
        mixup_ratio_t[idx] = mixups[index]
    loc_t[idx] = matched_gts  # [num_priors,4] encoded offsets to learn


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, offset=None, mixups=None, mixup_ratio_t=None):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # get prior_boxes
    if offset is not None:
        prior_boxes = priors.decode(offset[idx].detach()).squeeze(dim=0)
        offset[idx] = center_size(prior_boxes)
    else:
        prior_boxes = point_form(priors.prior_box)

    # jaccard index
    overlaps = box_iou(truths, prior_boxes)
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior

    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]          # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    if mixups is not None:
        mixup_ratio_t[idx] = mixups[best_truth_idx]
    loc_t[idx] = matches    # [num_priors,4] encoded offsets to learn


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

