from abc import ABCMeta, abstractmethod

import torch.nn as nn


class VM(nn.Module, metaclass=ABCMeta):
    """Vision Model Architecture (abstract)
    The network is composed of a pretrained backbone network followed by a normal layer
    for each level. The normalized features are given to the feature pyramid network (FPN)
    which combines the features of various levels. Finally, task-specific heads are
    conducted on the outputs of FPN to generate task prediction.

    Args:
        size: image size
        applications: tuple of target applications
        num_classes: num of classes of object detection and semantic segmentation
    """

    def __init__(self, size, applications, num_classes):
        super(VM, self).__init__()
        self.size = size
        self.num_classes = num_classes
        self.applications = applications

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_pretrained_params(self):
        pass

    @abstractmethod
    def get_from_scratch_params(self):
        pass

    @abstractmethod
    def get_last_shared_params(self):
        pass

    @staticmethod
    @abstractmethod
    def is_shared(name):
        pass

    def get_shared_params(self):
        return [named_param for named_param in self.named_parameters() if self.is_shared(named_param[0])]

    def get_task_specific_params(self):
        return [named_param for named_param in self.named_parameters() if not self.is_shared(named_param[0])]
