import tensorflow as tf
import numpy as np

from autokeras import utils
from autokeras.hypermodel import hyper_block as hb_module


class HyperPreprocessor(hb_module.HyperBlock):

    def build(self, hp, inputs=None):
        return inputs

    def fit_transform(self, hp, inputs):
        self.fit(hp, inputs)
        return self.transform(hp, inputs)

    def fit(self, hp, inputs):
        raise NotImplementedError

    def transform(self, hp, inputs):
        raise NotImplementedError


class OneHotEncoder(object):
    """A class that can format data.

    This class provides ways to transform data's classification label into
    vector.

    # Attributes
        data: The input data
        num_classes: The number of classes in the classification problem.
        labels: The number of labels.
        label_to_vec: Mapping from label to vector.
        int_to_label: Mapping from int to label.
    """

    def __init__(self):
        """Initialize a OneHotEncoder"""
        self.data = None
        self.num_classes = 0
        self.labels = None
        self.label_to_vec = {}
        self.int_to_label = {}

    def fit(self, data):
        """Create mapping from label to vector, and vector to label."""
        data = np.array(data).flatten()
        self.labels = set(data)
        self.num_classes = len(self.labels)
        for index, label in enumerate(self.labels):
            vec = np.array([0] * self.num_classes)
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
        return np.array(list(map(lambda x: self.int_to_label[x],
                                 np.argmax(np.array(data), axis=1))))


class Normalize(HyperPreprocessor):
    """ Perform basic image transformation and augmentation.

    # Attributes
        max_val: the maximum value of all data.
        mean: the mean value.
        std: the standard deviation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = None
        self.std = None

    def fit(self, hp, data):
        shape = utils.dataset_shape(data)
        axis = tuple(range(len(shape) - 1))

        def sum_up(old_state, new_elem):
            return old_state + new_elem

        def sum_up_sqaure(old_state, new_elem):
            return old_state + tf.square(new_elem)

        num_instance = data.reduce(np.float64(0), lambda x, _: x + 1)
        total_sum = data.reduce(np.float64(0), sum_up) / num_instance
        self.mean = tf.reduce_mean(total_sum, axis=axis)

        total_sum_square = data.reduce(np.float64(0), sum_up_sqaure) / num_instance
        square_mean = tf.reduce_mean(total_sum_square, axis=axis)
        self.std = tf.sqrt(square_mean - tf.square(self.mean))

    def transform(self, hp, data):
        """ Transform the test data, perform normalization.

        # Arguments
            data: Numpy array. The data to be transformed.

        # Returns
            A DataLoader instance.
        """
        # channel-wise normalize the image
        def normalize(x):
            return (x - self.mean) / self.std
        return data.map(normalize)
