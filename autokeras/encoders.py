import numpy as np
import tensorflow as tf

from autokeras.engine import encoder as encoder_module


class OneHotEncoder(encoder_module.Encoder):
    """OneHotEncoder to encode and decode the labels.

    This class provides ways to transform data's classification label into vector.

    # Arguments
        num_classes: The number of classes in the classification problem.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._label_to_vec = {}

    def get_config(self):
        config = super().get_config()
        config.update({'label_to_vec': self._label_to_vec})
        return config

    @classmethod
    def from_config(cls, config):
        obj = super().from_config(config)
        obj._label_to_vec = config['label_to_vec']

    def fit(self, data):
        """Create mapping from label to vector, and vector to label.

        # Arguments
            data: list or numpy.ndarray. The original labels.
        """
        data = np.array(data).flatten()
        self._labels = set(data)
        if not self.num_classes:
            self.num_classes = len(self._labels)
        if self.num_classes < len(self._labels):
            raise ValueError('More classes in data than specified.')
        for index, label in enumerate(self._labels):
            vec = np.array([0] * self.num_classes)
            vec[index] = 1
            self._label_to_vec[label] = vec
            self._int_to_label[index] = label

    def encode(self, data):
        """Get vector for every element in the data array.

        # Arguments
            data: list or numpy.ndarray. The original labels.

        # Returns
            numpy.ndarray. The one-hot encoded labels.
        """
        data = np.array(data)
        if len(data.shape) > 1:
            data = data.flatten()
        return np.array(list(map(lambda x: self._label_to_vec[x], data)))

    def decode(self, data):
        """Get label for every element in data.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The original labels.
        """
        return np.array(list(map(lambda x: self._int_to_label[x],
                                 np.argmax(np.array(data), axis=1)))).reshape(-1, 1)


class LabelEncoder(encoder_module.Encoder):
    """An encoder to encode the labels to integers.

    # Arguments
        num_classes: Int. The number of classes. Defaults to None.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._label_to_int = {}

    def get_config(self):
        config = super().get_config()
        config.update({'label_to_int': self._label_to_int})
        return config

    @classmethod
    def from_config(cls, config):
        obj = super().from_config(config)
        obj._label_to_int = config['label_to_int']

    def fit(self, data):
        """Fit the encoder with all the labels.

        # Arguments
            data: numpy.ndarray. The original labels.
        """
        data = np.array(data).flatten()
        self._labels = set(data)
        if not self.num_classes:
            self.num_classes = len(self._labels)
        if self.num_classes < len(self._labels):
            raise ValueError('More classes in data than specified.')
        for index, label in enumerate(self._labels):
            self._int_to_label[index] = label
            self._label_to_int[label] = index

    def transform(self, x):
        return self._label_to_int[x]

    def encode(self, data):
        """Encode the original labels.

        # Arguments
            data: numpy.ndarray. The original labels.

        # Returns
            numpy.ndarray with shape (n, 1). The encoded labels.
        """
        data = np.array(data)
        if len(data.shape) > 1:
            data = data.flatten()
        return np.array(list(map(lambda x: self._label_to_int[x],
                                 data))).reshape(-1, 1)

    def decode(self, data):
        """Get label for every element in data.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The original labels.
        """
        return np.array(list(map(lambda x: self._int_to_label[int(round(x[0]))],
                                 np.array(data)))).reshape(-1, 1)


def serialize(encoder):
    return tf.keras.utils.serialize_keras_object(encoder)


def deserialize(config, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='encoder')
