import tensorflow as tf
import sklearn
from tensorflow.python.util import nest

from autokeras import const


def get_global_average_pooling_layer_class(shape):
    return [tf.keras.layers.GlobalAveragePooling1D,
            tf.keras.layers.GlobalAveragePooling2D,
            tf.keras.layers.GlobalAveragePooling3D][len(shape) - 2]


def format_inputs(inputs, name=None, num=None):
    inputs = nest.flatten(inputs)
    if not isinstance(inputs, list):
        inputs = [inputs]

    if num is None:
        return inputs

    if not len(inputs) == num:
        raise ValueError('Expected {num} elements in the '
                         'inputs list for {name} '
                         'but received {len} inputs.'.format(num=num, name=name, len=len(inputs)))
    return inputs


def split_train_to_valid(x, y):
    # Generate split index
    validation_set_size = int(len(x[0]) * const.Constant.VALIDATION_SET_SIZE)
    validation_set_size = min(validation_set_size, 500)
    validation_set_size = max(validation_set_size, 1)
    train_index, valid_index = sklearn.model_selection.train_test_split(range(len(x[0])),
                                                                        test_size=validation_set_size,
                                                                        random_state=const.Constant.SEED)

    # Split the data
    x_train = None
    y_train = None
    x_val = None
    y_val = None
    for temp_x_train_input in x:
        x_train, x_val = temp_x_train_input[train_index], temp_x_train_input[valid_index]
    for temp_y_train_input in y:
        y_train, y_val = temp_y_train_input[train_index], temp_y_train_input[valid_index]

    return (x_train, y_train), (x_val, y_val)
