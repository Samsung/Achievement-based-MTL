import os
import pickle
import numpy as np
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast

from core.utils import Timer, Logger
from core.data.augmentations import TensorTransform
from layers.functions.detection import Detect
from core.metrics import intersection_and_union, evaluate_segmentation, compute_errors, evaluate_depth


@torch.no_grad()
def test_net(file_manager, net, test_loaders, loss_function=None, logger: Logger = None, iteration=None, ema=False):
    if logger is None:
        logger = print
    device = next(net.parameters()).device
    world_size, local_rank = (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)

    if loss_function is not None:
        losses = deepcopy(loss_function.losses)
        task_weights = loss_function.weight
        loss_function.clear()

    total_time, timer = Timer(), Timer()
    test_folder = file_manager.get_test_folder()
    det_file = os.path.join(test_folder, 'detections.pkl')

    results = {}
    depth_eval_result = torch.zeros(11, device=device)
    depth_metric = ['d1', 'd2', 'd3', 'rms', 'log_rms', 'abs_rel', 'sq_rel', 'si_log', 'log10']
    normal_error = torch.zeros(5, device=device)
    normal_metric = ['mean', 'median', '11.25', '22.5', '30']

    net.eval()
    net_without_ddp = net.module if hasattr(net, 'module') else net
    net_without_ddp = net_without_ddp.to(memory_format=torch.channels_last)
    net_without_ddp.apply(torch.quantization.disable_observer)

    transform = TensorTransform()

    total_time.tic()
    logger('test with batches of images on multi-GPU')
    for test_loader in test_loaders:
        # dump predictions and assoc. ground truth to text file for now
        collections = None
        testset = test_loader.dataset
        num_images = len(testset)
        num_classes = testset.num_classes
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        detector = Detect(num_classes, [0.1, 0.2], 400, 0.01, 0.5, 0.01, 200)
        seg_eval_result = torch.zeros([3, num_classes], device=device)

        timer.tic('data_loading')
        for i, sample in enumerate(test_loader, 1):
            with autocast():
                sample = {key: [anno.to(device) for anno in value] if type(value) is list else value.to(device)
                          for key, value in sample.items()}
                sample = transform(sample)
                timer.toc('data_loading')

                timer.tic('forward')
                preds = net(sample['images'].to(memory_format=torch.channels_last))
                timer.toc('forward')

                timer.tic('loss')
                if loss_function is not None:
                    loss_function(preds, sample)
                timer.toc('loss')

            timer.tic('post_proc')
            if 'detection' in preds and 'detection' in sample:
                priors = net_without_ddp.get_priors()
                detections = detector.detect(preds['detection'], priors)

                _, _, _h, _w = sample['images'].shape
                org_ratio = _w / _h
                for detection, h, w, index in zip(detections, sample['heights'], sample['widths'], sample['index']):
                    # skip j = 0, because it's the background class
                    for j in range(1, detection.size(0)):  # shape: [num_classes, top_k, 5]
                        dets = detection[j, :]  # shape: [top_k, 5]
                        dets = dets[dets[:, -1].gt(0)]
                        if dets.size(0) == 0:
                            continue
                        
                        dets[:, (0, 2)] *= _w
                        dets[:, (1, 3)] *= _h

                        sample_ratio = w / h
                        dets[:, :4] *= w / _w if sample_ratio > org_ratio else h / _h

                        if dist.is_initialized():
                            code = torch.tensor([[index, j]], dtype=torch.float32, device=device).expand(dets.shape[0], 2)
                            bboxes = torch.cat([code, dets], dim=1)
                            collections = bboxes if collections is None else torch.cat([bboxes, collections])
                        else:
                            all_boxes[j][index] = dets.cpu().numpy()

            if 'segmentation' in preds and 'segmentation' in sample:
                intersection, union, label = intersection_and_union(preds['segmentation'].max(1)[1],
                                                                    sample['segmentation'],
                                                                    num_classes, ignore_index=255)
                seg_eval_result += torch.stack([intersection, union, label])

            if 'depth' in preds and 'depth' in sample:
                depth_eval_result += compute_errors(preds, sample, testset.min_depth, testset.max_depth)

            if 'normal' in preds and 'normal' in sample:
                preds['normal'] = preds['normal'] / torch.norm(preds['normal'], p=2, dim=1, keepdim=True)
                for pred, gt in zip(preds['normal'], sample['normal']):
                    binary_mask = (gt.abs().sum(dim=0) != 0)
                    dot = (pred * gt).sum(dim=0).masked_select(binary_mask)

                    radian_error = torch.acos(torch.clamp(dot, -1, 1))
                    degree_error = torch.rad2deg(radian_error)

                    errors = torch.Tensor([
                        degree_error.mean(), degree_error.median(),
                        (degree_error < 11.25).sum(), (degree_error < 22.5).sum(), (degree_error < 30).sum()
                    ]).to(device)
                    errors[2:] /= degree_error.numel()
                    normal_error += errors / len(testset)
            timer.toc('post_proc', average=False)

            if i % 50 == 0:
                logger('im_detect: {:d}/{:d} {}'.format(i, len(test_loader), str(timer)))
                timer.clear()
            timer.tic('data_loading')

        # Assign collection into all boxes
        if dist.is_initialized():
            if 'detection' in preds and 'detection' in sample and collections is not None:
                # Obtain the number of detections of all GPUs
                num_bboxes = torch.tensor([collections.shape[0]], dtype=torch.int32).cuda()
                num_list = [torch.ones_like(num_bboxes) for _ in range(world_size)]
                dist.all_gather(num_list, num_bboxes)

                # Initialize collection_list with zero
                collections_list = list()
                for i, _num_bboxes in enumerate(num_list):
                    collections_list.append(collections if i == local_rank
                                            else torch.zeros([_num_bboxes, collections.shape[1]], device=device))
                collections_list = torch.cat(collections_list)

                # Collect distributed results
                dist.all_reduce(collections_list, op=dist.ReduceOp.SUM)
                collections = collections_list

                # Assign collections into all_boxes
                collections = collections.cpu().numpy()
                for collection in collections:
                    img_idx, cls_idx, bboxes = int(collection[0]), int(collection[1]), np.expand_dims(collection[2:7], axis=0)
                    if len(all_boxes[cls_idx][img_idx]):
                        all_boxes[cls_idx][img_idx] = np.vstack((all_boxes[cls_idx][img_idx], bboxes))
                    else:
                        all_boxes[cls_idx][img_idx] = bboxes

            if 'segmentation' in preds and 'segmentation' in sample:
                dist.all_reduce(seg_eval_result, dist.reduce_op.SUM)

            if 'depth' in preds and 'depth' in sample:
                dist.all_reduce(depth_eval_result, dist.reduce_op.SUM)

            if 'normal' in preds and 'normal' in sample:
                dist.all_reduce(normal_error, dist.reduce_op.SUM)

        if not dist.is_initialized() or not local_rank:
            if 'detection' in preds and 'detection' in sample:
                logger('Evaluating detections')
                with open(det_file, 'wb') as f:
                    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
                APs, mAP = testset.evaluate_detections(all_boxes, test_folder)
                results['mAP_%s' % testset.year if hasattr(testset, 'year') else 'mAP'] = mAP[0]  # IoU = 0.50:0.95
                APs = {key: value for key, value in zip(testset.columns, APs)}
                if isinstance(logger, Logger):
                    logger.to_csv(iteration, str(testset), APs)
            if 'segmentation' in preds and 'segmentation' in sample:
                logger('Evaluating segmentations')
                iou, mAcc, allACC = evaluate_segmentation(seg_eval_result)
                results['mIoU'] = iou.mean().item()

            if 'depth' in preds and 'depth' in sample:
                logger('Evaluating depths')
                depth_eval_result = evaluate_depth(depth_eval_result)
                results = dict(results, **depth_eval_result)
            if 'normal' in preds and 'normal' in sample:
                logger('Evaluating surface normal')
                results['mean'] = normal_error[0].item()
                results['median'] = normal_error[1].item()
                results['11.25'] = normal_error[2].item()
                results['22.5'] = normal_error[3].item()
                results['30'] = normal_error[4].item()
        else:
            if 'detection' in preds and 'detection' in sample:
                results['mAP_%s' % testset.year if hasattr(testset, 'year') else 'mAP'] = 0
            if 'segmentation' in preds and 'segmentation' in sample: results['mIoU'] = 0
            if 'depth' in preds and 'depth' in sample: results = dict(results, **dict.fromkeys(depth_metric, 0))
            if 'normal' in preds and 'normal' in sample: results = dict(results, **dict.fromkeys(normal_metric, 0))

    # Broadcast mAP from rank 0 to all GPUs
    if dist.is_initialized():
        accuracy = torch.tensor([value for value in results.values()], dtype=torch.float32, device='cuda') \
            if not local_rank else torch.tensor([0 for _ in results.keys()], dtype=torch.float32, device='cuda')
        dist.broadcast(accuracy, 0)
        results = {key: accuracy[i].item() for i, key in enumerate(results.keys())}

    # combine losses and evaluation metrics
    summary = {**loss_function.items(), **results} if loss_function else {**results}
    if isinstance(logger, Logger):
        logger.write_summary(iteration, summary, train=False, ema=ema)
    logger(''.join(['%s: %3.4f ' % (key, value) for key, value in summary.items()]))

    if loss_function:
        loss_function.losses = losses
        loss_function.weight = task_weights
        if hasattr(loss_function, 'update_kpi'):
            loss_function.update_kpi(results)

    net.train()
    net_without_ddp.apply(torch.quantization.enable_observer)

    eval_time = total_time.toc()
    logger('total evaluation time is {:.3f}'.format(eval_time))
    return results
