"""VOC Dataset Classes
Original author: Yi Lu
https://github.com/luuuyi/RefineDet.PyTorch/blob/master/data/voc0712.py
Updated by: Hanjoo cho (hanjoo.cho@samsung.com) and Hayoung Yun (hayoung.yun@samsung.com)
"""

import os
import os.path as osp
import torch.utils.data as data
import cv2
import numpy as np
import pickle
from PIL import Image
from .voc_eval import voc_eval
import xml.etree.ElementTree as ET

VOC_CLASSES = ['__background__',    # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initialized with a dictionary lookup of class names to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of class names -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable will be an ET.Element
        Returns:
            a list containing lists of bounding boxes in percent coordinate  [bbox coords, class name]
        """

        height = int(target.find('size').find('height').text.lower())
        width = int(target.find('size').find('width').text.lower())

        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_sets (string): image_set to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, root, train=True, transform=None,
                 image_sets=(('2007', 'trainval'), ('2012', 'trainval')),
                 target_transform=VOCAnnotationTransform()):
        self.name = 'VOC0712'
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform

        self._img_path = osp.join('%s', 'JPEGImages', '%s.jpg')
        self._ann_path = osp.join('%s', 'Annotations', '%s.xml')
        self._label_path = osp.join('%s', 'SegmentationClass', '%s.png') if not train \
            else osp.join('%s', 'SegmentationClassAug', '%s.png')

        self.num_classes = len(VOC_CLASSES)
        self.columns = VOC_CLASSES[1:] + ['mAP@50:95', 'mAP@50', 'mAP@75']

        self.index = list()
        for (year, name) in image_sets:
            self.year = year
            root_path = osp.join(self.root, 'VOC' + year)
            self.index += [(root_path, line.strip()) for line
                           in open(osp.join(root_path, 'ImageSets', 'Main', name + '.txt'))]

    def __len__(self):
        return len(self.index)

    def __str__(self):
        return ', '.join(['%s_%s' % (name, year) for year, name in self.image_set]) + '_in_%s' % self.name

    def __getitem__(self, index):
        idx = self.index[index]
        image = self.get_image(idx)
        target = self.get_target(idx)
        label_map = self.get_label_map(idx)

        height, width, _ = image.shape
        sample = {'index': index, 'image': image, 'height': height, 'width': width}
        if target is not None:
            sample['target'] = target
        if label_map is not None:
            sample['label_map'] = label_map
        return self.transform(sample)

    def get_image(self, idx):
        image = cv2.imread(self._img_path % idx, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_target(self, idx):
        target = ET.parse(self._ann_path % idx).getroot()
        target = np.array(self.target_transform(target))
        if not target.shape[0]:
            target = None
        return target

    def get_label_map(self, idx):
        label_path = self._label_path % idx
        label_map = np.array(Image.open(label_path)).astype(np.uint8) if os.path.exists(label_path) else None
        return label_map

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.
        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes, output_dir)
        aps, map = self._do_python_eval(output_dir)
        return aps, map

    def _get_voc_results_file_template(self, output_dir=None):
        if output_dir is None:
            output_dir = self.root

        filename = 'comp4_det_test' + '_{:s}.txt'
        file_dir = os.path.join(output_dir, 'results', 'VOC' + self.year, 'Main')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        path = os.path.join(file_dir, filename)
        return path

    def _write_voc_results_file(self, all_boxes, output_dir):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            cls_ind = cls_ind
            if cls == '__background__':
                continue
            # print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template(output_dir).format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.index):
                    year, img_id = index[0][-4:], index[1]
                    img_id = f'{year}_{img_id}' if img_id[:2] != '20' else img_id
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(img_id, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        cachedir = os.path.join(self.root, 'annotations_cache')

        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self.year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        aps = np.zeros((len(VOC_CLASSES) - 1, 10))
        for i, cls in enumerate(VOC_CLASSES):
            if cls == '__background__':
                continue

            image_names, anno_names = self.imageset_to_image_path()
            filename = self._get_voc_results_file_template(output_dir).format(cls)
            rec, prec, ap = voc_eval(filename, anno_names, image_names, cls, cachedir, use_07_metric)
            aps[i - 1] = ap
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        maps = [np.mean(aps), np.mean(aps[:, 0]), np.mean(aps[:, 5])]
        aps = list(aps[:, 0]) + maps

        print('Results:')
        print('\t'.join(VOC_CLASSES[1:] + ['mAP@50:95', 'mAP@50', 'mAP@75']))
        print('\t'.join(['%.4f' % ap for ap in aps]))
        return aps, maps

    def imageset_to_image_path(self):
        image_names, anno_files = [], []
        for image_set in self.image_set:
            year, name = image_set[0], image_set[1]
            root_path = os.path.join(self.root, 'VOC' + year)
            anno_path = os.path.join(root_path, 'Annotations', '{:s}.xml')
            imageset_file = os.path.join(root_path, 'ImageSets', 'Main', name + '.txt')

            with open(imageset_file, 'r') as f:
                lines = f.readlines()
            anno_files = anno_files + [anno_path.format(x.strip()) for x in lines]
            image_names = image_names + [x.strip() if x[0:2] == '20' else f'{year}_{x.strip()}' for x in lines]
        return image_names, anno_files
