from functools import reduce
from operator import mul

import numpy as np
from keras.layers import Dense, Conv1D, Conv2D, Conv3D

from autokeras.utils import get_conv_layer_func


def conv_to_deeper_layer(conv_layer):
    filter_shape = conv_layer.kernel_size
    n_filters = conv_layer.filters
    weight = np.zeros(filter_shape + (n_filters, n_filters))
    center = ((filter_shape[0] - 1) / 2, (filter_shape[1] - 1) / 2)
    for i in range(n_filters):
        filter_weight = np.zeros(filter_shape + (n_filters,))
        filter_weight[center[0], center[1], i] = 1
        weight[:, :, :, i] = filter_weight
    bias = np.zeros(n_filters)
    conv_func = get_conv_layer_func(len(filter_shape))
    new_conv_layer = conv_func(n_filters, kernel_size=filter_shape, activation='relu')
    new_conv_layer.build((None,) * len(filter_shape) + (n_filters,))
    new_conv_layer.set_weights((weight, bias))
    return new_conv_layer


def dense_to_deeper_layer(dense_layer):
    units = dense_layer.units
    weight = np.eye(units)
    bias = np.zeros(units)
    new_dense_layer = Dense(units, activation='relu')
    new_dense_layer.build((None, units))
    new_dense_layer.set_weights((weight, bias))
    return new_dense_layer


def to_deeper_layer(layer):
    if isinstance(layer, Dense):
        return dense_to_deeper_layer(layer)
    elif isinstance(layer, (Conv1D, Conv2D, Conv3D)):
        return conv_to_deeper_layer(layer)
    raise ValueError("Layer must be a Dense or Convolution Layer")


def dense_to_wider_layer(pre_layer, next_layer, n_add_units):
    n_units1 = pre_layer.get_weights().shape[0]
    n_units2 = pre_layer.get_weights().shape[1]
    n_units3 = next_layer.get_weights().shape[1]

    teacher_w1 = pre_layer.get_weights()[0]
    teacher_b1 = pre_layer.get_weights()[1]
    teacher_w2 = next_layer.get_weights()[0]
    teacher_b2 = next_layer.get_weights()[1]
    rand = np.random.randint(n_units2, size=n_add_units)
    replication_factor = np.bincount(rand)
    student_w1 = teacher_w1.copy()
    student_w2 = teacher_w2.copy()
    student_b1 = teacher_b1.copy()

    # target layer update (i)
    for i in range(n_add_units):
        teacher_index = rand[i]
        new_weight = teacher_w1[:, teacher_index]
        new_weight = new_weight[:, np.newaxis]
        student_w1 = np.concatenate((student_w1, new_weight), axis=1)
        student_b1 = np.append(student_b1, teacher_b1[teacher_index])

    # next layer update (i+1)
    for i in range(n_add_units):
        teacher_index = rand[i]
        n_copies = replication_factor[teacher_index] + 1
        new_weight = teacher_w2[teacher_index, :]*(1./n_copies)
        new_weight = new_weight[np.newaxis, :]
        student_w2 = np.concatenate((student_w2, new_weight), axis=0)
        student_w2[teacher_index, :] = new_weight

    new_pre_layer = Dense(n_units2)
    new_pre_layer.build((None, n_units1))
    new_pre_layer.set_weights((student_w1, student_b1))
    new_next_layer = Dense(n_units3)
    new_next_layer.build((None, n_units2))
    new_next_layer.set_weights((student_w2, teacher_b2))

    return new_pre_layer, new_next_layer


def conv_to_wider_layer(pre_layer, next_layer, n_add_filters):
    pre_filter_shape = pre_layer.kernel_size
    next_filter_shape = next_layer.kernel_size
    conv_func = get_conv_layer_func(len(pre_filter_shape))
    n_pre_filters = pre_layer.filters
    n_next_filters = next_layer.filters

    teacher_w1 = pre_layer.get_weights()[0]
    teacher_w2 = next_layer.get_weights()[0]
    teacher_b1 = pre_layer.get_weights()[1]
    teacher_b2 = next_layer.get_weights()[1]
    rand = np.random.randint(n_pre_filters, size=n_add_filters)
    replication_factor = np.bincount(rand)
    student_w1 = teacher_w1.copy()
    student_w2 = teacher_w2.copy()
    student_b1 = teacher_b1.copy()
    # target layer update (i)
    for i in range(len(rand)):
        teacher_index = rand[i]
        new_weight = teacher_w1[:, :, :, teacher_index]
        new_weight = new_weight[:, :, :, np.newaxis]
        student_w1 = np.concatenate((student_w1, new_weight), axis=3)
        student_b1 = np.append(student_b1, teacher_b1[teacher_index])
    # next layer update (i+1)
    for i in range(len(rand)):
        teacher_index = rand[i]
        factor = replication_factor[teacher_index] + 1
        new_weight = teacher_w2[:, :, teacher_index, :] * (1. / factor)
        new_weight_re = new_weight[:, :, np.newaxis, :]
        student_w2 = np.concatenate((student_w2, new_weight_re), axis=2)
        student_w2[:, :, teacher_index, :] = new_weight

    new_pre_layer = conv_func(n_pre_filters, kernel_size=pre_filter_shape, activation='relu')
    new_pre_layer.build((None,) * len(pre_filter_shape) + (pre_filter_shape[-1],))
    new_pre_layer.set_weights((student_w1, student_b1))
    new_next_layer = conv_func(n_next_filters, kernel_size=next_filter_shape, activation='relu')
    new_next_layer.build((None,) * len(next_filter_shape) + (next_filter_shape[-1],))
    new_next_layer.set_weights((student_w2, teacher_b2))
    return new_pre_layer, new_next_layer


def conv_dense_to_wider_layer(pre_layer, next_layer, n_add_filters):
    pre_filter_shape = pre_layer.kernel_size
    conv_func = get_conv_layer_func(len(pre_filter_shape))
    n_pre_filters = pre_layer.filters
    n_units = next_layer.get_weights().shape[1]

    teacher_w1 = pre_layer.get_weights()[0]
    teacher_w2 = next_layer.get_weights()[0]
    n_total_weights = reduce(mul, teacher_w2.shape)
    teacher_w2 = teacher_w2.reshape(n_total_weights / n_pre_filters / n_units, n_pre_filters, n_units)
    teacher_b1 = pre_layer.get_weights()[1]
    teacher_b2 = next_layer.get_weights()[1]
    rand = np.random.randint(n_pre_filters, size=n_add_filters)
    replication_factor = np.bincount(rand)
    student_w1 = teacher_w1.copy()
    student_w2 = teacher_w2.copy()
    student_b1 = teacher_b1.copy()
    # target layer update (i)
    for i in range(len(rand)):
        teacher_index = rand[i]
        new_weight = teacher_w1[:, :, :, teacher_index]
        new_weight = new_weight[:, :, :, np.newaxis]
        student_w1 = np.concatenate((student_w1, new_weight), axis=3)
        student_b1 = np.append(student_b1, teacher_b1[teacher_index])
    # next layer update (i+1)
    for i in range(len(rand)):
        teacher_index = rand[i]
        factor = replication_factor[teacher_index] + 1
        new_weight = teacher_w2[:, teacher_index, :] * (1. / factor)
        new_weight_re = new_weight[:, np.newaxis, :]
        student_w2 = np.concatenate((student_w2, new_weight_re), axis=1)
        student_w2[:, teacher_index, :] = new_weight

    new_pre_layer = conv_func(n_pre_filters, kernel_size=pre_filter_shape, activation='relu')
    new_pre_layer.build((None,) * len(pre_filter_shape) + (pre_filter_shape[-1],))
    new_pre_layer.set_weights((student_w1, student_b1))
    new_next_layer = Dense(n_units)
    new_next_layer.build((None, n_units))
    new_next_layer.set_weights((student_w2.flatten(), teacher_b2))
    return new_pre_layer, new_next_layer


def to_wider_layer(pre_layer, next_layer, n_add):
    if isinstance(pre_layer, (Conv1D, Conv2D, Conv3D)) and isinstance(next_layer, Dense):
        return conv_dense_to_wider_layer(pre_layer, next_layer, n_add)
    elif isinstance(pre_layer, Dense) and isinstance(next_layer, Dense):
        return dense_to_wider_layer(pre_layer, next_layer, n_add)
    elif isinstance(pre_layer, (Conv1D, Conv2D, Conv3D)) and isinstance(next_layer, (Conv1D, Conv2D, Conv3D)):
        return conv_to_wider_layer(pre_layer, next_layer, n_add)
    raise ValueError("Layer must be a Dense or Convolution Layer")
