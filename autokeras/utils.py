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
    return tf.data.Dataset.zip(new_x)


def prepare_preprocess(x, y=None, validation_data=None):
    """Convert each input to a tf.data.Dataset."""
    x = inputs_to_datasets(x)
    if not y:
        return x, None
    if y:
        y = inputs_to_datasets(y)
    return x, y


def prepare_model_input(x=None, y=None, validation_data=None, batch_size=32):
    """Zip multiple tf.data.Dataset into one Dataset."""
    if not y:
        return tf.data.Dataset.zip(tuple(x)).batch(batch_size), None
    if not validation_data:
        return tf.data.Dataset.zip((tuple(x), tuple(y))).batch(batch_size), None
    return tf.data.Dataset.zip(
        (tuple(x), tuple(y))).batch(batch_size), tf.data.Dataset.zip(
        (tuple(validation_data[0]),
         tuple(validation_data[1]))).batch(batch_size)


def is_label(y):
    return len(y.flatten()) == len(y) and len(set(y.flatten())) > 2
