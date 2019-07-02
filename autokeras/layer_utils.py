import tensorflow as tf
from tensorflow.python.util import nest


def get_global_average_pooling_layer(shape):
    return [tf.keras.layers.GlobalAveragePooling1D,
            tf.keras.layers.GlobalAveragePooling2D,
            tf.keras.layers.GlobalAveragePooling3D][len(shape) - 3]


def get_global_max_pooling_layer(shape):
    return [tf.keras.layers.GlobalMaxPool1D,
            tf.keras.layers.GlobalMaxPool2D,
            tf.keras.layers.GlobalMaxPool3D][len(shape) - 3]


def format_inputs(inputs, name=None, num=None):
    inputs = nest.flatten(inputs)
    if not isinstance(inputs, list):
        inputs = [inputs]

    if num is None:
        return inputs

    if not len(inputs) == num:
        raise ValueError('Expected {num} elements in the '
                         'inputs list for {name} '
                         'but received {len} inputs.'.format(num=num,
                                                             name=name,
                                                             len=len(inputs)))
    return inputs


def split_train_to_valid(x, y, validation_split):
    # Generate split index
    validation_set_size = int(len(x[0]) * validation_split)

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

