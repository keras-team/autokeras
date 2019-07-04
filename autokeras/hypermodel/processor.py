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
        mean: Tensor. The mean value. Shape: (data last dimension length,)
        std: Tensor. The standard deviation. Shape is the same as mean.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = None
        self.std = None

    def fit(self, hp, inputs):
        shape = utils.dataset_shape(inputs)
        axis = tuple(range(len(shape) - 1))

        def sum_up(old_state, new_elem):
            return old_state + new_elem

        def sum_up_sqaure(old_state, new_elem):
            return old_state + tf.square(new_elem)

        num_instance = inputs.reduce(np.float64(0), lambda x, _: x + 1)
        total_sum = inputs.reduce(np.float64(0), sum_up) / num_instance
        self.mean = tf.reduce_mean(total_sum, axis=axis)

        total_sum_square = inputs.reduce(np.float64(0), sum_up_sqaure) / num_instance
        square_mean = tf.reduce_mean(total_sum_square, axis=axis)
        self.std = tf.sqrt(square_mean - tf.square(self.mean))

    def transform(self, hp, inputs):
        """ Transform the test data, perform normalization.

        # Arguments
            data: Tensorflow Dataset. The data to be transformed.

        # Returns
            A DataLoader instance.
        """

        # channel-wise normalize the image
        def normalize(x):
            return (x - self.mean) / self.std

        return inputs.map(normalize)


class Tokenize(HyperPreprocessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()

    def fit(self, hp, inputs):
        def fit_on_texts(x):
            self.tokenizer.fit_on_texts(x)
        inputs.map(fit_on_texts)

    def transform(self, hp, inputs):
        def texts_to_sequences(x):
            self.tokenizer.texts_to_sequences([x])
        return inputs.map(texts_to_sequences)
