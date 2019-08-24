import pickle
import re

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest


def get_global_average_pooling(shape):
    return [tf.keras.layers.GlobalAveragePooling1D,
            tf.keras.layers.GlobalAveragePooling2D,
            tf.keras.layers.GlobalAveragePooling3D][len(shape) - 3]


def get_global_max_pooling(shape):
    return [tf.keras.layers.GlobalMaxPool1D,
            tf.keras.layers.GlobalMaxPool2D,
            tf.keras.layers.GlobalMaxPool3D][len(shape) - 3]


def get_max_pooling(shape):
    return [tf.keras.layers.MaxPool1D,
            tf.keras.layers.MaxPool2D,
            tf.keras.layers.MaxPool3D][len(shape) - 3]


def get_conv(shape):
    return [tf.keras.layers.Conv1D,
            tf.keras.layers.Conv2D,
            tf.keras.layers.Conv3D][len(shape) - 3]


def get_sep_conv(shape):
    return [tf.keras.layers.SeparableConv1D,
            tf.keras.layers.SeparableConv2D,
            tf.keras.layers.Conv3D][len(shape) - 3]


def get_dropout(shape):
    return [tf.keras.layers.SpatialDropout1D,
            tf.keras.layers.SpatialDropout2D,
            tf.keras.layers.SpatialDropout3D][len(shape) - 3]


def validate_num_inputs(inputs, num):
    inputs = nest.flatten(inputs)
    if not len(inputs) == num:
        raise ValueError('Expected {num} elements in the inputs list '
                         'but received {len} inputs.'.format(num=num,
                                                             len=len(inputs)))


def split_train_to_valid(x, y, validation_split):
    # Generate split index
    validation_set_size = int(len(x[0]) * validation_split)
    validation_set_size = max(validation_set_size, 1)
    validation_set_size = min(validation_set_size, len(x[0]) - 1)

    # Split the data
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    for temp_x in x:
        x_train.append(temp_x[:-validation_set_size])
        x_val.append(temp_x[-validation_set_size:])
    for temp_y in y:
        y_train.append(temp_y[:-validation_set_size])
        y_val.append(temp_y[-validation_set_size:])

    return (x_train, y_train), (x_val, y_val)


def get_name_scope():
    with tf.name_scope('a') as scope:
        name_scope = scope[:-2]
    return name_scope


def dataset_shape(dataset):
    return tf.compat.v1.data.get_output_shapes(dataset)


def inputs_to_datasets(x):
    x = nest.flatten(x)
    new_x = []
    for temp_x in x:
        if isinstance(temp_x, np.ndarray):
            new_x.append(tf.data.Dataset.from_tensor_slices(temp_x))
    return tf.data.Dataset.zip(tuple(new_x))


def prepare_preprocess(x, y):
    """Convert each input to a tf.data.Dataset."""
    x = inputs_to_datasets(x)
    y = inputs_to_datasets(y)
    return tf.data.Dataset.zip((x, y))


def is_label(y):
    """Check if the targets are one-hot encoded or plain labels.

    # Arguments
        y: numpy.ndarray. The targets.

    # Returns
        Boolean. Whether the targets are plain label, not encoded.
    """
    return len(y.flatten()) == len(y) and len(set(y.flatten())) > 2


def pickle_from_file(path):
    """Load the pickle file from the provided path and returns the object."""
    return pickle.load(open(path, 'rb'))


def pickle_to_file(obj, path):
    """Save the pickle file to the specified path."""
    pickle.dump(obj, open(path, 'wb'))


def to_snake_case(name):
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure


class OneHotEncoder(object):
    """A class that can format data.

    This class provides ways to transform data's classification label into vector.

    # Arguments
        num_classes: The number of classes in the classification problem.
    """

    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self._labels = None
        self._label_to_vec = {}
        self._int_to_label = {}

    def fit_with_labels(self, data):
        """Create mapping from label to vector, and vector to label.

        # Arguments
            data: list or numpy.ndarray. The labels.
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

    def fit_with_one_hot_encoded(self, data):
        """Create mapping from label to vector, and vector to label from one-hot.

        # Arguments
            data: numpy.ndarray. The one-hot encoded labels.
        """
        data = np.array(data)
        if not self.num_classes:
            self.num_classes = data.shape[0]
        self._labels = set(range(self.num_classes))
        for label in self._labels:
            vec = np.array([0] * self.num_classes)
            vec[label] = 1
            self._label_to_vec[label] = vec
            self._int_to_label[label] = label

    def encode(self, data):
        """Get vector for every element in the data array.

        # Arguments
            data: list or numpy.ndarray. The labels.
        """
        data = np.array(data)
        if len(data.shape) > 1:
            data = data.flatten()
        return np.array(list(map(lambda x: self._label_to_vec[x], data)))

    def decode(self, data):
        """Get label for every element in data.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.
        """
        return np.array(list(map(lambda x: self._int_to_label[x],
                                 np.argmax(np.array(data), axis=1))))
