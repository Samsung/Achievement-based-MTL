"""COCO Dataset Classes

Original authors: Ross Girshick and Xinlei Chen
https://github.com/dd604/refinedet.pytorch/blob/master/libs/datasets/coco.py

Updated by: Hanjoo.Cho (hanjoo.cho@samsung.com) and Hayoung Yun (hayoung.yun@samsung.com)
"""

import json
import pickle

import cv2
import numpy as np
import os
import os.path
import torch.utils.data as data
from PIL import Image

import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCODetection(data.Dataset):
    """MS-COCO Detection Dataset
    Arguments:
        root (string): file path to COCO dataset folder.
        image_set (string): image_set to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the input image
    """

    def __init__(self, root, image_set, transform=None):
        self.root_dir = root
        self.image_set = image_set
        self.transform = transform

        self._setup()
        self.columns = list(self._classes)[1:] + ['AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                                                  'AP @[ IoU=0.50      | area=   all | maxDets=100 ]',
                                                  'AP @[ IoU=0.75      | area=   all | maxDets=100 ]',
                                                  'AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                                                  'AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                                                  'AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                                                  'AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                                                  'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                                                  'AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                                                  'AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                                                  'AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                                                  'AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]']

    def _setup(self):
        self.name = 'COCO'
        self.image_path = '{}images/{}/%012d.jpg'.format(self.root_dir, self.image_set)
        self.label_path = '{}/stuff/{}/%012d.png'.format(self.root_dir, self.image_set)
        self._setup_class_configs()

    def __len__(self):
        return len(self.index)

    def __str__(self):
        return '%s_in_%s' % (self.image_set, self.name)

    def __getitem__(self, index):
        img = self.get_image(index)
        height, width, _ = img.shape

        label_map = self.get_label_map(index)
        target = self.get_target(index, height, width)

        sample = {'index': index, 'image': img, 'height': height, 'width': width}
        if target is not None:
            sample['target'] = target
        if label_map is not None:
            sample['label_map'] = label_map

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_image(self, index):
        img_path = self.image_path % self.index[index]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_label_map(self, index):
        label_path = self.label_path % self.index[index]
        label_map = np.array(Image.open(label_path)).astype(np.uint8) if os.path.exists(label_path) else None

        if label_map is not None:
            label_map = self.lut[label_map]

        return label_map

    def get_target(self, index, height, width):
        target = self._annotation_from_index(self.index[index])
        target = np.array(target)

        if target.shape[0] == 0:
            return None

        target[:, (0, 2)] /= width
        target[:, (1, 3)] /= height
        return target

    def evaluate_detections(self, all_boxes, output_dir):
        try:
            res_file = os.path.join(output_dir, ('detections_' + self.name + '_results'))
            res_file += '.json'
            self._write_coco_results_file(all_boxes, res_file)

            # Only do evaluation on non-test sets
            if self.name.find('test') == -1:
                aps, map = self._do_detection_eval(res_file, output_dir)
                return aps + list(map), map
            return [0] * len(self.columns), [0] * 12
        except IndexError:  # empty box
            return [0] * len(self.columns), [0] * 12

    def _setup_class_configs(self):
        self.coco_tool = COCO(self._get_ann_file(self.image_set))

        self._set_detection_config()
        self.index = self.coco_tool.getImgIds()

        # setup for semantic segmentation
        self.num_classes = 171
        self.coco_cat_id_to_class_ind.update({i: i - 10 for i in range(91, 183)})
        lut = [255 for i in range(256)]
        for key, value in self.coco_cat_id_to_class_ind.items():
            lut[key] = value
        self.lut = np.array(lut)

    def _set_detection_config(self):
        categories = self.coco_tool.loadCats(self.coco_tool.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in categories])
        self.num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        self._class_to_coco_cat_id = dict(zip([c['name'] for c in categories], self.coco_tool.getCatIds()))
        self.coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls], self._class_to_ind[cls])
                                              for cls in self._classes[1:]])

    def _get_ann_file(self, name):
        prefix = 'instances' if name.find('test') == -1 else 'image_info'
        return os.path.join(self.root_dir, 'annotations', prefix + '_' + name + '.json')

    def _annotation_from_index(self, index):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = self.coco_tool.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco_tool.getAnnIds(imgIds=index, iscrowd=None)
        objs = self.coco_tool.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2]))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3]))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            res[ix, 0:4] = obj['clean_bbox']
            res[ix, 4] = cls

        return res

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        aps = []

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            aps.append(100 * ap)

        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()
        return aps, coco_eval.stats

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        coco_dt = self.coco_tool.loadRes(res_file)
        coco_eval = COCOeval(self.coco_tool, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        aps, map = self._print_detection_eval_metrics(coco_eval)
        eval_file = os.path.join(output_dir, 'detection_results.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))
        return aps, map

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.index):
            dets = boxes[im_ind]
            if len(dets) == 0:
                continue
            dets = dets.astype(np.float_)
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs
            hs = dets[:, 3] - ys
            results.extend(
                [{'image_id': index,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind], coco_cat_id))
            '''
            if cls_ind ==30:
                res_f = res_file+ '_1.json'
                print('Writing results json to {}'.format(res_f))
                with open(res_f, 'w') as fid:
                    json.dump(results, fid)
                results = []
            '''
        # res_f2 = res_file+'_2.json'
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)
