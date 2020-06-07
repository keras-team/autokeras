import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest


def batched(dataset):
    shape = nest.flatten(dataset_shape(dataset))[0]
    return len(shape) > 0 and shape[0] is None


def batch_dataset(dataset, batch_size):
    if batched(dataset):
        return dataset
    return dataset.batch(batch_size)


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
                         'batches to be split.')
    validation_set_size = min(
        max(int(num_instances * validation_split), 1),
        num_instances - 1)
    train_set_size = num_instances - validation_set_size
    train_dataset = dataset.take(train_set_size)
    validation_dataset = dataset.skip(train_set_size)
    return train_dataset, validation_dataset


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
