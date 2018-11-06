import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose
from abc import ABC, abstractmethod
from autokeras.constant import Constant
from autokeras.utils import read_csv_file, read_image


class OneHotEncoder:
    """A class that can format data.

    This class provides ways to transform data's classification label into vector.

    Attributes:
          data: The input data
          n_classes: The number of classes in the classification problem.
          labels: The number of labels.
          label_to_vec: Mapping from label to vector.
          int_to_label: Mapping from int to label.
    """

    def __init__(self):
        """Initialize a OneHotEncoder"""
        self.data = None
        self.n_classes = 0
        self.labels = None
        self.label_to_vec = {}
        self.int_to_label = {}

    def fit(self, data):
        """Create mapping from label to vector, and vector to label."""
        data = np.array(data).flatten()
        self.labels = set(data)
        self.n_classes = len(self.labels)
        for index, label in enumerate(self.labels):
            vec = np.array([0] * self.n_classes)
            vec[index] = 1
            self.label_to_vec[label] = vec
            self.int_to_label[index] = label

    def transform(self, data):
        """Get vector for every element in the data array."""
        data = np.array(data)
        if len(data.shape) > 1:
            data = data.flatten()
        return np.array(list(map(lambda x: self.label_to_vec[x], data)))

    def inverse_transform(self, data):
        """Get label for every element in data."""
        return np.array(list(map(lambda x: self.int_to_label[x], np.argmax(np.array(data), axis=1))))


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class DataTransformer(ABC):
    @abstractmethod
    def transform_train(self, data, targets=None, batch_size=None):
        raise NotImplementedError

    @abstractmethod
    def transform_test(self, data, target=None, batch_size=None):
        raise NotImplementedError


class TextDataTransformer(DataTransformer):

    def transform_train(self, data, targets=None, batch_size=None):
        dataset = self._transform(compose_list=[], data=data, targets=targets)
        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(data), batch_size)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def transform_test(self, data, targets=None, batch_size=None):
        dataset = self._transform(compose_list=[], data=data, targets=targets)
        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(data), batch_size)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _transform(self, compose_list, data, targets):
        data = torch.Tensor(data.transpose(0, 3, 1, 2))
        data_transforms = Compose(compose_list)
        return MultiTransformDataset(data, targets, data_transforms)


class ImageDataTransformer(DataTransformer):
    def __init__(self, data, augment=Constant.DATA_AUGMENTATION):
        self.max_val = data.max()
        data = data / self.max_val
        self.mean = np.mean(data, axis=(0, 1, 2), keepdims=True).flatten()
        self.std = np.std(data, axis=(0, 1, 2), keepdims=True).flatten()
        self.augment = augment

    def transform_train(self, data, targets=None, batch_size=None):
        short_edge_length = min(data.shape[1], data.shape[2])
        common_list = [Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))]
        if self.augment:
            compose_list = [ToPILImage(),
                            RandomCrop(data.shape[1:3], padding=4),
                            RandomHorizontalFlip(),
                            ToTensor()
                            ] + common_list + [Cutout(n_holes=Constant.CUTOUT_HOLES,
                                                      length=int(short_edge_length * Constant.CUTOUT_RATIO))]
        else:
            compose_list = common_list

        if len(data.shape) != 4:
            compose_list = []

        dataset = self._transform(compose_list, data, targets)

        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(data), batch_size)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def transform_test(self, data, targets=None, batch_size=None):
        common_list = [Normalize(torch.Tensor(self.mean), torch.Tensor(self.std))]
        compose_list = common_list
        if len(data.shape) != 4:
            compose_list = []

        dataset = self._transform(compose_list, data, targets)

        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(data), batch_size)

        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def _transform(self, compose_list, data, targets):
        data = data / self.max_val
        args = [0, len(data.shape) - 1] + list(range(1, len(data.shape) - 1))
        data = torch.Tensor(data.transpose(*args))
        data_transforms = Compose(compose_list)
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


class BatchDataset(Dataset):
    def __init__(self, csv_file_path, image_path, has_target=True):
        file_names, target = read_csv_file(csv_file_path)

        self.y_encoder = OneHotEncoder()
        self.y_encoder.fit(target)
        target = self.y_encoder.transform(target)

        self.target = target
        self.has_target = has_target
        self.file_paths = list(map(lambda file_name: os.path.join(image_path, file_name), file_names))

    def __getitem__(self, index):
        image = read_image(self.file_paths[index])
        if len(image.shape) < 3:
            image = image[..., np.newaxis]
        image = torch.Tensor(image.transpose(2, 0, 1))
        if self.has_target:
            return image, self.target[index]
        return image

    def __len__(self):
        return len(self.file_paths)
