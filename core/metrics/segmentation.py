import torch


def intersection_and_union(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output[target == ignore_index] = ignore_index
    output = output[output != ignore_index]
    target = target[target != ignore_index]
    intersection = output[output == target]

    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def evaluate_segmentation(seg_eval_result):
    intersection, union, label = seg_eval_result
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (label + 1e-10)
    # mIoU = iou_class[1:].mean()  # excluding background
    mAcc = torch.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(label) + 1e-10)
    return iou_class, mAcc, allAcc
