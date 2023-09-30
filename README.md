# Achievement-based Training Progress Balancing for Multi-Task Learning

This is the official implementation of ICCV'23 paper [**Achievement-based Training Progress Balancing for Multi-Task Learning**]() by Hayoung Yun and Hanjoo Cho [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yun_Achievement-Based_Training_Progress_Balancing_for_Multi-Task_Learning_ICCV_2023_paper.pdf)[[Video]]()

![image](https://github.com/Samsung/Achievement-based-MTL/assets/24874629/3adc7ec6-1ddb-4f14-b740-b23ab0e76a53)

In this paper, we address two major challenges of multi-task learning (1) (2).

We propose an achievement-based multi-task loss to modulate training speed of various task based on their accuracy ”achievement,” defined as the ratio of current accuracy to singletask accuracy. We formulate the multi-task loss as a weighted geometric mean of individual task losses instead of a weighted sum to prevent any task from dominating the loss.



The proposed loss achieved the best multi-task accuracy without incurring training time overhead. Compared to singletask models, the proposed one achieved 1.28%, 1.65%, and 1.18% accuracy improvement in object detection, semantic segmentation, and depth estimation, respectively, while reducing computations to 33.73%.


## Contents
0. [Requirements](#requirements)
0. [Installation](#installation)
0. [Datasets](#datasets)
0. [Training](#training)
0. [Citation](#citation)

## Requirements
- Python3.8
- CUDA 11.3 
- PyTorch 1.13

## Installation
### Clone this repository.   
```
git clone https://github.com/Samsung/Achievement-based-MTL.git
```
### Install requirements
```
pip install -r requirements.txt
```

## Datasets
We support PASCAL VOC and NYU v2 datasets now.
Download and organize the dataset files as follows:

### VOC Dataset
```Shell
$datasets/VOC/
```

### NYU v2 Dataset
```Shell
$datasets/NYU/
```

## Experiments 

### Benchmark Methods

### Scripts

#### Training using the conventional fully-annotated Multi-Dataset (NYUv2)
```
python3 train_test.py cfg/detection/VOC/VMM_efficientnet-v2-s.cfg
python3 train_test.py cfg/segmentation/VOC/VMM_efficientnet-v2-s.cfg
python3 train_test.py cfg/depth/VOC/VMM_efficientnet-v2-s.cfg
...
```

#### Training using the partially-annotated multi-dataset (PASCAL VOC + NYU depth)

```
python3 train_test.py cfg/detection/VOC/VMM_efficientnet-v2-s.cfg
python3 train_test.py cfg/segmentation/VOC/VMM_efficientnet-v2-s.cfg
python3 train_test.py cfg/depth/VOC/VMM_efficientnet-v2-s.cfg
...
```

## Citation
```
@InProceedings{Yun_2023_ICCV,
    author    = {Yun, Hayoung and Cho, Hanjoo},
    title     = {Achievement-Based Training Progress Balancing for Multi-Task Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16935-16944}
}
```
If you have any questions, please feel free to contact Hayoung YUN(hayoung.yun@samsung.com) and Hanjoo CHO(hanjoo.cho@samsung.com)
