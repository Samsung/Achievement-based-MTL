# Achievement-based Training Progress Balancing for Multi-Task Learning

This is the official implementation of ICCV'23 paper **Achievement-based Training Progress Balancing for Multi-Task Learning** by Hayoung Yun and Hanjoo Cho [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yun_Achievement-Based_Training_Progress_Balancing_for_Multi-Task_Learning_ICCV_2023_paper.pdf)[[Video]](https://github.com/Samsung/Achievement-based-MTL/assets/24874629/a0f8d899-0185-426d-be18-5affe3fc8391)[[Poster]](https://github.com/Samsung/Achievement-based-MTL/files/12775627/hanjoo_cho_achievement-based_training_progress_balancing_for_multi-task_learning_iccv_2023.pdf)

In this paper, we address two major challenges of multi-task learning (1) the high cost of annotating labels for all tasks and (2) balancing training progress of diverse tasks with distinct characteristics. 

![image](https://github.com/Samsung/Achievement-based-MTL/assets/24874629/2beb52b8-c727-46f0-b3a5-f396648ba700)

We address the high annotation cost by integrating task-specific datasets to construct a large-scale multi-task datset. The composed dataset is thereby partially-annotated because each image of the dataset is labeled only for the task from which it originated. Hence, the numbers of labels for individual tasks could be different. The difference in the number of task labels exacerbates the imbalance in training progress among tasks. To handle the intensified imbalance, we propose a novel multi-task loss named achievement-based multi-task loss.  

![image](https://github.com/Samsung/Achievement-based-MTL/assets/24874629/650ad209-4660-40fc-8076-cdc9d4d73b46)

The previous accuracy-based multi-task loss, DTP, focused on the current accuracy of each task. Instead, we pay attention to how the accuracy can be improved further. For that, considering the accuracy of the single-task model as the accuracy _potential_ of the task, we define an ”_achievement_” as the ratio of current accuracy to its potential. Our achievement-based task weights encourage tasks with low achievements and slow down tasks converged early. 

Then, we formulate a multi-task loss as weighted geometric mean, instead of a weighted sum generally used for multi-task losses. A weighted sum can be easily dominated by the largest one, if their scales are significantly different. Hence, we employ the weighted geometric mean to multi-task loss to capture the variance in all losses. 

![image](https://github.com/Samsung/Achievement-based-MTL/assets/24874629/71ec91e2-dba6-4f81-913b-51133e9b0bea)

The proposed loss achieved the best multi-task accuracy without incurring training time overhead. Compared to single-task models, the proposed one achieved 1.28%, 1.65%, and 1.18% accuracy improvement in object detection, semantic segmentation, and depth estimation, respectively, while reducing computations to 33.73%.

## Contents
1. [Installation](#installation)
1. [Datasets](#datasets)
1. [Experiments](#experiments)
1. [Citation](#citation)

## Installation

### Our setup
- Python3.8
- CUDA 11.3 
- PyTorch 1.13

### Script

#### Clone this repository.   
```
git clone https://github.com/Samsung/Achievement-based-MTL.git
```
#### Install requirements
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

### Supported Multi-Task Losses

Method | Paper
--| --
RLW (rlw) | --
DWA	(dwa) |
GLS	(geometric) |
MGDA (mgda) |
PCGrad (pcgrad) |
CAGrad (cagrad) |
GradNorm (grad-norm) |
IMTL (imtl / imtl-g) |
DTP	(dtp) |
Proposed (amtl) |


### Scripts

#### Training using the conventional fully-annotated Multi-Dataset (NYUv2)
```
# single-task
python3 train_test.py cfg/segmentation/NYU/DeepLab_resnet50.cfg
python3 train_test.py cfg/depth/NYU/DeepLab_resnet50.cfg
python3 train_test.py cfg/normal/NYU/DeepLab_resnet50.cfg

# multi-task
python3 train_test.py cfg/seg+depth+normal/NYU/DeepLab_resnet50.cfg
```

#### Training using the partially-annotated multi-dataset (PASCAL VOC + NYU depth)

```
# single-task
python3 train_test.py cfg/detection/VOC/VMM_efficientnet-v2-s.cfg
python3 train_test.py cfg/segmentation/VOC/VMM_efficientnet-v2-s.cfg
python3 train_test.py cfg/depth/NYU/VMM_efficientnet-v2-s.cfg

# multi-task
python3 train_test.py cfg/seg+det+depth/NYU/VMM_efficientnet-v2-s.cfg
```

## Citation
```reference
@InProceedings{Yun_2023_ICCV,
    author    = {Yun, Hayoung and Cho, Hanjoo},
    title     = {Achievement-Based Training Progress Balancing for Multi-Task Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16935-16944}
}
```
If you have any questions, please feel free to contact Hayoung Yun (hayoung.yun@samsung.com) and Hanjoo Cho (hanjoo.cho@samsung.com)
