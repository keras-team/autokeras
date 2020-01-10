import pickle
import re

import numpy as np
import tensorflow as tf
from packaging.version import parse
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


def split_dataset(dataset, validation_split):
    """Split dataset into training and validation.

    # Arguments
        dataset: tf.data.Dataset. The entire dataset to be split.
        validation_split: Float. The split ratio for the validation set.

    # Raises
        ValueError: If the dataset provided is too small to be split.

    # Returns
        A tuple of two tf.data.Dataset. The training set and the validation set.
    """
    num_instances = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    if num_instances < 2:
        raise ValueError('The dataset should at least contain 2 '
                         'instances to be split.')
    validation_set_size = min(
        max(int(num_instances * validation_split), 1),
        num_instances - 1)
    train_set_size = num_instances - validation_set_size
    train_dataset = dataset.take(train_set_size)
    validation_dataset = dataset.skip(train_set_size)
    return train_dataset, validation_dataset


def get_name_scope():
    with tf.name_scope('a') as scope:
        name_scope = scope[:-2]
    return name_scope


def dataset_shape(dataset):
    return tf.compat.v1.data.get_output_shapes(dataset)


def is_label(y):
    """Check if the targets are one-hot encoded or plain labels.

    # Arguments
        y: numpy.ndarray. The targets.

    # Returns
        Boolean. Whether the targets are plain label, not encoded.
    """
    return len(y.flatten()) == len(y)


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


def to_type_key(dictionary, convert_func):
    """Convert the keys of a dictionary to a different type.

    # Arguments
        dictionary: Dictionary. The dictionary to be converted.
        convert_func: Function. The function to convert a key.
    """
    return {convert_func(key): value
            for key, value in dictionary.items()}


def check_tf_version():
    if parse(tf.__version__) < parse('2.0.0'):
        raise ImportError(
            f'The Tensorflow package version needs to be at least v2.0.0 \n'
            f'for AutoKeras to run. Currently, your TensorFlow version is \n'
            f'v{tf.__version__}. Please upgrade with \n'
            f'`$ pip install --upgrade tensorflow` -> GPU version \n'
            f'or \n'
            f'`$ pip install --upgrade tensorflow-cpu` -> CPU version. \n'
            f'You can use `pip freeze` to check afterwards that everything is ok.'
        )
