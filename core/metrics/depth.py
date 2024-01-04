import torch
import torch.nn.functional as F
import numpy as np


def cls_label_to_depth(preds, beta, shift, depth_class):
    _, _, H, W = preds['depth'].shape
    depth_label = preds['depth'].view(-1, 2, depth_class, H, W)
    depth_label = torch.unsqueeze((torch.sum((F.softmax(depth_label, dim=1)[:, 0, :, :, :] > 0.5), dim=1) - 1), dim=1)

    t0 = torch.exp(np.log(beta) * depth_label / depth_class)
    t1 = torch.exp(np.log(beta) * (depth_label + 1) / depth_class)
    depth_label = (t0 + t1) / 2 - shift
    preds['depth'] = depth_label


def compute_errors(preds, sample, min_depth, max_depth):
    # Legacy code for depth as ordinal classification
    # if use_ord_cls:
    #     cls_label_to_depth(preds, loss_function)

    preds['depth'][preds['depth'] < min_depth] = min_depth
    preds['depth'][preds['depth'] > max_depth] = max_depth
    preds['depth'][torch.isinf(preds['depth'])] = max_depth
    preds['depth'][torch.isnan(preds['depth'])] = min_depth
    valid_mask = torch.logical_and(sample['depth'] > min_depth, sample['depth'] < max_depth)

    gt, pred = sample['depth'][valid_mask], preds['depth'][valid_mask]

    numel = gt.numel()
    thresh = torch.max((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).float().sum()
    d2 = (thresh < 1.25 ** 2).float().sum()
    d3 = (thresh < 1.25 ** 3).float().sum()

    rms = torch.sum((gt - pred) ** 2)
    log_rms = torch.sum((torch.log(gt) - torch.log(pred)) ** 2)

    abs_rel = torch.sum(torch.abs(gt - pred) / gt)
    sq_rel = torch.sum(((gt - pred) ** 2) / gt)

    err = torch.log(pred) - torch.log(gt)
    err_sum = torch.sum(err)
    err_sq_sum = torch.sum(err ** 2)

    err = torch.abs(torch.log10(pred) - torch.log10(gt))
    log10 = torch.sum(err)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.Tensor([numel, err_sum, err_sq_sum, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]).to(device)


def evaluate_depth(depth_eval_result):
    results = dict()
    [numel, err_sum, err_sq_sum, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3] = depth_eval_result

    results['d1'] = (d1 / numel).item()
    results['d2'] = (d2 / numel).item()
    results['d3'] = (d3 / numel).item()

    results['rms'] = torch.sqrt(rms / numel).item()
    results['log_rms'] = torch.sqrt(log_rms / numel).item()

    results['abs_rel'] = (abs_rel / numel).item()
    results['sq_rel'] = (sq_rel / numel).item()

    results['si_log'] = torch.sqrt((err_sq_sum / numel) - (err_sum / numel) ** 2).item()
    results['log10'] = (log10 / numel).item()
    return results

