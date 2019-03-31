from abc import ABC, abstractmethod

import numpy as np


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


class DataTransformer(ABC):
    """A superclass for all the DataTransformer."""

    def __init__(self):
        pass

    @abstractmethod
    def transform_train(self, data, targets=None, batch_size=None):
        """ Transform the training data and get the DataLoader class.

        Args:
            data: x.
            targets: y.
            batch_size: the batch size.

        Returns:
            dataloader: A torch.DataLoader class to represent the transformed data.
        """
        raise NotImplementedError

    @abstractmethod
    def transform_test(self, data, targets=None, batch_size=None):
        """ Transform the training data and get the DataLoader class.

        Args:
            data: x.
            targets: y.
            batch_size: the batch size.

        Returns:
            dataloader: A torch.DataLoader class to represent the transformed data.
        """
        raise NotImplementedError
