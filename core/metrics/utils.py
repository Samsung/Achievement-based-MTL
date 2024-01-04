higher_better = ('mAP', 'mAP_2007', 'mAP_2012', 'mIoU', 'd1', '11.25', '22.5', '30')
lower_better = ('rms', 'log_rms', 'mean', 'median')
detection_metrics = ('mAP', 'mAP_2007', 'mAP_2012')
segmentation_metrics = ('mIoU',)
depth_metrics = ('d1', 'rms')
normal_metrics = ('mean', 'median', '11.25')


def merge_metrics(results: dict):
    if not results:
        return 0
    eps = 1e-10
    metrics = dict()
    num_metrics = dict()
    for key, value in results.items():
        if key in detection_metrics:
            if 'det' not in metrics:
                metrics['det'] = 1
                num_metrics['det'] = 0
            num_metrics['det'] += 1
            metrics['det'] *= value
        elif key in segmentation_metrics:
            if 'seg' not in metrics:
                metrics['seg'] = 1
                num_metrics['seg'] = 0
            num_metrics['seg'] += 1
            metrics['seg'] *= value
        elif key in depth_metrics:
            if 'depth' not in metrics:
                metrics['depth'] = 1
                num_metrics['depth'] = 0
            if key in higher_better:
                num_metrics['depth'] += 1
                metrics['depth'] *= value
            elif key in lower_better:
                num_metrics['depth'] += 1
                metrics['depth'] *= 1 / (value + eps)
        elif key in normal_metrics:
            if 'normal' not in metrics:
                metrics['normal'] = 1
                num_metrics['normal'] = 0
            if key in higher_better:
                num_metrics['normal'] += 1
                metrics['normal'] *= value
            elif key in lower_better:
                num_metrics['normal'] += 1
                metrics['normal'] *= 1 / (value + eps)

    total_metric = 1
    num_tasks = 0
    for key, value in metrics.items():
        num_tasks += 1
        total_metric *= (value ** (1 / num_metrics[key]))
    total_metric = total_metric ** (1 / num_tasks)
    return total_metric
