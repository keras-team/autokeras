import numpy as np

from autokeras import constant
from autokeras.layers import StubConv, StubBatchNormalization, StubActivation, StubDropout, StubDense, \
    StubWeightedAdd

NOISE_RATIO = 1e-4


def deeper_conv_block(conv_layer, kernel_size, weighted=True):
    """Get deeper layer for convolution layer

    Args:
        weighted:
        conv_layer: the convolution layer from which we get deeper layer
        kernel_size: the size of kernel

    Returns:
        The deeper convolution layer
    """
    filter_shape = (kernel_size,) * (len(conv_layer.kernel_size))
    n_filters = conv_layer.filters
    weight = np.zeros(filter_shape + (n_filters, n_filters))
    center = tuple(map(lambda x: int((x - 1) / 2), filter_shape))
    for i in range(n_filters):
        filter_weight = np.zeros(filter_shape + (n_filters,))
        index = center + (i,)
        filter_weight[index] = 1
        weight[..., i] = filter_weight
    bias = np.zeros(n_filters)
    new_conv_layer = StubConv(n_filters, kernel_size=filter_shape, func=conv_layer.func)
    bn = StubBatchNormalization()

    if weighted:
        new_conv_layer.set_weights((add_noise(weight, np.array([0, 1])), add_noise(bias, np.array([0, 1]))))
        new_weights = [np.ones(n_filters, dtype=np.float32),
                       np.zeros(n_filters, dtype=np.float32),
                       np.zeros(n_filters, dtype=np.float32),
                       np.ones(n_filters, dtype=np.float32)]
        bn.set_weights(new_weights)

    return [new_conv_layer,
            bn,
            StubActivation('relu'),
            StubDropout(constant.CONV_DROPOUT_RATE)]


def dense_to_deeper_block(dense_layer, weighted=True):
    """Get deeper layer for dense layer

    Args:
        weighted:
        dense_layer: the dense layer from which we get deeper layer

    Returns:
        The deeper dense layer
    """
    units = dense_layer.units
    weight = np.eye(units)
    bias = np.zeros(units)
    new_dense_layer = StubDense(units, dense_layer.activation)
    if weighted:
        new_dense_layer.set_weights((add_noise(weight, np.array([0, 1])), add_noise(bias, np.array([0, 1]))))
    return [new_dense_layer, StubDropout(constant.DENSE_DROPOUT_RATE)]


def wider_pre_dense(layer, n_add, weighted=True):
    """Get previous dense layer for current layer

   Args:
       weighted:
       layer: the layer from which we get wide previous dense layer
       n_add: output shape

   Returns:
       The previous dense layer
   """
    if not weighted:
        return StubDense(layer.units + n_add, layer.activation)

    n_units2 = layer.units

    teacher_w, teacher_b = layer.get_weights()
    rand = np.random.randint(n_units2, size=n_add)
    student_w = teacher_w.copy()
    student_b = teacher_b.copy()

    # target layer update (i)
    for i in range(n_add):
        teacher_index = rand[i]
        new_weight = teacher_w[:, teacher_index]
        new_weight = new_weight[:, np.newaxis]
        student_w = np.concatenate((student_w, add_noise(new_weight, student_w)), axis=1)
        student_b = np.append(student_b, add_noise(teacher_b[teacher_index], student_b))

    new_pre_layer = StubDense(n_units2 + n_add, layer.activation)
    new_pre_layer.set_weights((student_w, student_b))

    return new_pre_layer


def wider_pre_conv(layer, n_add_filters, weighted=True):
    """Get previous convolution layer for current layer

   Args:
       weighted:
       layer: layer from which we get wider previous convolution layer
       n_add_filters: the filters size of convolution layer

   Returns:
       The previous convolution layer
   """
    if not weighted:
        return StubConv(layer.filters + n_add_filters, kernel_size=layer.kernel_size, func=layer.func)

    pre_filter_shape = layer.kernel_size
    n_pre_filters = layer.filters
    rand = np.random.randint(n_pre_filters, size=n_add_filters)
    teacher_w, teacher_b = layer.get_weights()
    student_w = teacher_w.copy()
    student_b = teacher_b.copy()
    # target layer update (i)
    for i in range(len(rand)):
        teacher_index = rand[i]
        new_weight = teacher_w[..., teacher_index]
        new_weight = new_weight[..., np.newaxis]
        student_w = np.concatenate((student_w, new_weight), axis=-1)
        student_b = np.append(student_b, teacher_b[teacher_index])
    new_pre_layer = StubConv(n_pre_filters + n_add_filters, kernel_size=pre_filter_shape, func=layer.func)
    new_pre_layer.set_weights((add_noise(student_w, teacher_w), add_noise(student_b, teacher_b)))
    return new_pre_layer


def wider_next_conv(layer, start_dim, total_dim, n_add, weighted=True):
    """Get next wider convolution layer for current layer

    Args:
       weighted:
       layer: the layer from which we get wider next convolution layer
       start_dim: the started dimension
       total_dim: the total dimension
       n_add: the filters size of convolution layer

    Returns:
       The next wider convolution layer
    """
    if not weighted:
        return StubConv(layer.filters, kernel_size=layer.kernel_size, func=layer.func)
    filter_shape = layer.kernel_size
    n_filters = layer.filters
    teacher_w, teacher_b = layer.get_weights()

    new_weight_shape = list(teacher_w.shape)
    new_weight_shape[-2] = n_add
    new_weight = np.zeros(tuple(new_weight_shape))

    student_w = np.concatenate((teacher_w[..., :start_dim, :].copy(),
                                add_noise(new_weight, teacher_w),
                                teacher_w[..., start_dim:total_dim, :].copy()), axis=-2)
    new_layer = StubConv(n_filters, kernel_size=filter_shape, func=layer.func)
    new_layer.set_weights((student_w, teacher_b))
    return new_layer


def wider_bn(layer, start_dim, total_dim, n_add, weighted=True):
    """Get new layer with wider batch normalization for current layer

   Args:
       weighted:
       layer: the layer from which we get new layer with wider batch normalization
       start_dim: the started dimension
       total_dim: the total dimension
       n_add: the output shape

   Returns:
       The new layer with wider batch normalization
   """
    if not weighted:
        return StubBatchNormalization()

    weights = layer.get_weights()

    new_weights = [np.ones(n_add, dtype=np.float32),
                   np.zeros(n_add, dtype=np.float32),
                   np.zeros(n_add, dtype=np.float32),
                   np.ones(n_add, dtype=np.float32)]

    student_w = tuple()
    for weight, new_weight in zip(weights, new_weights):
        temp_w = weight.copy()
        temp_w = np.concatenate((temp_w[:start_dim], new_weight, temp_w[start_dim:total_dim]))
        student_w += (temp_w,)
    new_layer = StubBatchNormalization()
    new_layer.set_weights(student_w)
    return new_layer


def wider_next_dense(layer, start_dim, total_dim, n_add, weighted=True):
    """Get next dense layer for current layer

    Args:
       weighted:
       layer: the dense layer from which we search next dense layer
       n_add: output shape
       start_dim: the started dimension
       total_dim: the total dimension

    Returns:
       The next dense layer
    """
    if not weighted:
        return StubDense(layer.units, layer.activation)
    n_units = layer.units
    teacher_w, teacher_b = layer.get_weights()
    student_w = teacher_w.copy()
    n_units_each_channel = int(teacher_w.shape[0] / total_dim)

    new_weight = np.zeros((n_add * n_units_each_channel, teacher_w.shape[1]))
    student_w = np.concatenate((student_w[:start_dim * n_units_each_channel],
                                add_noise(new_weight, student_w),
                                student_w[start_dim * n_units_each_channel:total_dim * n_units_each_channel]))

    new_layer = StubDense(n_units, layer.activation)
    new_layer.set_weights((student_w, teacher_b))
    return new_layer


def wider_weighted_add(layer, n_add, weighted=True):
    """Return wider weighted add layer

    Args:
        weighted:
        layer: the layer from which we get wider weighted add layer
        n_add: output shape

    Returns:
        The wider weighted add layer
    """
    if not weighted:
        return StubWeightedAdd()

    n_add += 0
    new_layer = StubWeightedAdd()
    new_layer.set_weights(layer.get_weights())
    return new_layer


def add_noise(weights, other_weights):
    w_range = np.ptp(other_weights.flatten())
    noise_range = NOISE_RATIO * w_range
    noise = np.random.uniform(-noise_range / 2.0, noise_range / 2.0, weights.shape)
    return np.add(noise, weights)
