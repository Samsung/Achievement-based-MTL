# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import pickle
import xml.etree.ElementTree as ET

import numpy as np
import os


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(det_path, anno_names, image_names, class_name, cache_dir, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath, anno_names, image_names, class_name, [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    det_path: Path to detections
        detpath.format(classname) should produce the detection results file.
    anno_names: Path to the xml annotations file
    image_names: Text file containing the list of images, one image per line.
    class_name: Category name (duh)
    cache_dir: Directory for caching the annotations
    [use_07_metric]: Whether to use VOC07's 11 point AP computation (default False)
    """

    # first load gt
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    cache_file = os.path.join(cache_dir, 'voc_annots.pkl')
    if not os.path.isfile(cache_file):
        # load annots
        recs = {}
        for i, (image_name, anno_name) in enumerate(zip(image_names, anno_names)):
            recs[image_name] = parse_rec(anno_name)
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(image_names)))
        # save
        print('Saving cached annotations to {:s}'.format(cache_file))
        with open(cache_file, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cache_file, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for image_name in image_names:
        R = [obj for obj in recs[image_name] if obj['name'] == class_name]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool_)
        det = [[False] * 10 for _ in range(len(R))]
        npos = npos + sum(~difficult)
        class_recs[image_name] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    # read dets
    det_file = det_path.format(class_name)
    with open(det_file, 'r') as f:
        lines = f.readlines()

        if len(lines) == 0:
            return 0, 0, 0  # if there is no detection, it return 0, 0, 0

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros((nd, 10))
    fp = np.zeros((nd, 10))
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        ov_base = np.arange(0.5, 1, 0.05)
        for i in range(10):
            if ovmax > ov_base[i]:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax][i]:
                        tp[d][i] = 1.
                        R['det'][jmax][i] = True
                    else:
                        fp[d][i] = 1.
            else:
                fp[d][i] = 1.

    # compute precision recall
    fp = fp.transpose().cumsum(axis=1)
    tp = tp.transpose().cumsum(axis=1)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    aps = [voc_ap(rec[i], prec[i], use_07_metric) for i in range(10)]
    return rec[0], prec[0], aps
