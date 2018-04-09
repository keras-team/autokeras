import numpy as np


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
