import torch

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose

from autokeras.constant import Constant


class OneHotEncoder:
    """A class that can format data

    This class provide ways to transform data's classification into vector

    Attributes:
          data: the input data
          n_classes: the number of classification
          labels: the number of label
          label_to_vec: mapping from label to vector
          int_to_label: mapping from int to label
    """

    def __init__(self):
        """Init OneHotEncoder"""
        self.data = None
        self.n_classes = 0
        self.labels = None
        self.label_to_vec = {}
        self.int_to_label = {}

    def fit(self, data):
        """Create mapping from label to vector, and vector to label"""
        data = np.array(data).flatten()
        self.labels = set(data)
        self.n_classes = len(self.labels)
        for index, label in enumerate(self.labels):
            vec = np.array([0] * self.n_classes)
            vec[index] = 1
            self.label_to_vec[label] = vec
            self.int_to_label[index] = label

    def transform(self, data):
        """Get vector for every element in the data array"""
        data = np.array(data)
        if len(data.shape) > 1:
            data = data.flatten()
        return np.array(list(map(lambda x: self.label_to_vec[x], data)))

    def inverse_transform(self, data):
        """Get label for every element in data"""
        return np.array(list(map(lambda x: self.int_to_label[x], np.argmax(np.array(data), axis=1))))


class DataTransformer:
    def __init__(self, data, augment=Constant.DATA_AUGMENTATION):
        self.max_val = data.max()
        data = data / self.max_val
        self.mean = np.mean(data, axis=(0, 1, 2), keepdims=True).flatten()
        self.std = np.std(data, axis=(0, 1, 2), keepdims=True).flatten()
        self.augment = augment

    def transform_train(self, data, targets=None):
        data = data / self.max_val
        data = torch.Tensor(data.transpose(0, 3, 1, 2))
        if not self.augment:
            augment_list = []
        else:
            augment_list = [ToPILImage(),
                            RandomCrop(data.shape[2:], padding=4),
                            RandomHorizontalFlip(),
                            ToTensor()
                            ]
        common_list = [Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))]
        data_transforms = Compose(augment_list + common_list)
        return MultiTransformDataset(data, targets, data_transforms)

    def transform_test(self, data, targets=None):
        data = data / self.max_val
        data = torch.Tensor(data.transpose(0, 3, 1, 2))
        common_list = [Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))]
        data_transforms = Compose(common_list)
        return MultiTransformDataset(data, targets, data_transforms)


class MultiTransformDataset(Dataset):
    def __init__(self, dataset, target, compose):
        self.dataset = dataset
        self.target = target
        self.compose = compose

    def __getitem__(self, index):
        feature = self.dataset[index]
        if self.target is None:
            return self.compose(feature)
        return self.compose(feature), self.target[index]

    def __len__(self):
        return len(self.dataset)
