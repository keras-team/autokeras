import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize, ToPILImage, RandomCrop, RandomHorizontalFlip, ToTensor, Compose

from autokeras.constant import Constant
from autokeras.preprocessor import DataTransformer


class ImageDataTransformer(DataTransformer):
    """ Perform basic image transformation and augmentation.

    Attributes:
        max_val: the maximum value of all data.
        mean: the mean value.
        std: the standard deviation.
        augment: whether to perform augmentation on data.
    """

    def __init__(self, data, augment=None):
        super().__init__()
        self.max_val = data.max()
        data = data / self.max_val
        self.mean = np.mean(data, axis=(0, 1, 2), keepdims=True).flatten()
        self.std = np.std(data, axis=(0, 1, 2), keepdims=True).flatten()
        if augment is None:
            self.augment = Constant.DATA_AUGMENTATION
        else:
            self.augment = augment

    def transform_train(self, data, targets=None, batch_size=None):
        """ Transform the training data, perform random cropping data augmentation and basic random flip augmentation.

        Args:
            data: Numpy array. The data to be transformed.
            batch_size: int batch_size.
            targets: the target of training set.

        Returns:
            A DataLoader class instance.
        """
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
        """ Transform the test data, perform normalization.

        Args:
            data: Numpy array. The data to be transformed.
            batch_size: int batch_size.
            targets: the target of test set.
        Returns:
            A DataLoader instance.
        """
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
        """Perform the actual transformation.

        Args:
            compose_list: a list of transforming operation.
            data: x.
            targets: y.

        Returns:
            A MultiTransformDataset class to represent the dataset.
        """
        data = data / self.max_val
        args = [0, len(data.shape) - 1] + list(range(1, len(data.shape) - 1))
        data = torch.Tensor(data.transpose(*args))
        data_transforms = Compose(compose_list)
        return MultiTransformDataset(data, targets, data_transforms)


class MultiTransformDataset(Dataset):
    """A class incorporate all transform method into a torch.Dataset class."""

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
        """Perform the actual transformation.

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


class DataTransformerMlp(DataTransformer):
    def __init__(self, data):
        super().__init__()
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform_train(self, data, targets=None, batch_size=None):
        data = (data - self.mean) / self.std
        data = np.nan_to_num(data)
        dataset = self._transform([], data, targets)

        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(data), batch_size)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def transform_test(self, data, target=None, batch_size=None):
        return self.transform_train(data, targets=target, batch_size=batch_size)

    @staticmethod
    def _transform(compose_list, data, targets):
        args = [0, len(data.shape) - 1] + list(range(1, len(data.shape) - 1))
        data = torch.Tensor(data.transpose(*args))
        data_transforms = Compose(compose_list)
        return MultiTransformDataset(data, targets, data_transforms)