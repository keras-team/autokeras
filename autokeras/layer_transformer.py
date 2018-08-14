import numpy as np

from autokeras.layers import StubConv, StubBatchNormalization, StubDense, StubReLU

NOISE_RATIO = 1e-4


def deeper_conv_block(conv_layer, kernel_size, weighted=True):
    filter_shape = (kernel_size,) * 2
    n_filters = conv_layer.filters
    weight = np.zeros((n_filters, n_filters) + filter_shape)
    center = tuple(map(lambda x: int((x - 1) / 2), filter_shape))
    for i in range(n_filters):
        filter_weight = np.zeros((n_filters,) + filter_shape)
        index = (i,) + center
        filter_weight[index] = 1
        weight[i, ...] = filter_weight
    bias = np.zeros(n_filters)
    new_conv_layer = StubConv(conv_layer.filters, n_filters, kernel_size=kernel_size)
    bn = StubBatchNormalization(n_filters)

    if weighted:
        new_conv_layer.set_weights((add_noise(weight, np.array([0, 1])), add_noise(bias, np.array([0, 1]))))
        new_weights = [add_noise(np.ones(n_filters, dtype=np.float32), np.array([0, 1])),
                       add_noise(np.zeros(n_filters, dtype=np.float32), np.array([0, 1])),
                       add_noise(np.zeros(n_filters, dtype=np.float32), np.array([0, 1])),
                       add_noise(np.ones(n_filters, dtype=np.float32), np.array([0, 1]))]
        bn.set_weights(new_weights)

    return [StubReLU(),
            new_conv_layer,
            bn]


def dense_to_deeper_block(dense_layer, weighted=True):
    units = dense_layer.units
    weight = np.eye(units)
    bias = np.zeros(units)
    new_dense_layer = StubDense(units, units)
    if weighted:
        new_dense_layer.set_weights((add_noise(weight, np.array([0, 1])), add_noise(bias, np.array([0, 1]))))
    return [StubReLU(), new_dense_layer]


def wider_pre_dense(layer, n_add, weighted=True):
    if not weighted:
        return StubDense(layer.input_units, layer.units + n_add)

    n_units2 = layer.units

    teacher_w, teacher_b = layer.get_weights()
    rand = np.random.randint(n_units2, size=n_add)
    student_w = teacher_w.copy()
    student_b = teacher_b.copy()

    # target layer update (i)
    for i in range(n_add):
        teacher_index = rand[i]
        new_weight = teacher_w[teacher_index, :]
        new_weight = new_weight[np.newaxis, :]
        student_w = np.concatenate((student_w, add_noise(new_weight, student_w)), axis=0)
        student_b = np.append(student_b, add_noise(teacher_b[teacher_index], student_b))

    new_pre_layer = StubDense(layer.input_units, n_units2 + n_add)
    new_pre_layer.set_weights((student_w, student_b))

    return new_pre_layer


def wider_pre_conv(layer, n_add_filters, weighted=True):
    if not weighted:
        return StubConv(layer.input_channel, layer.filters + n_add_filters, kernel_size=layer.kernel_size)

    n_pre_filters = layer.filters
    rand = np.random.randint(n_pre_filters, size=n_add_filters)
    teacher_w, teacher_b = layer.get_weights()

    student_w = teacher_w.copy()
    student_b = teacher_b.copy()
    # target layer update (i)
    for i in range(len(rand)):
        teacher_index = rand[i]
        new_weight = teacher_w[teacher_index, ...]
        new_weight = new_weight[np.newaxis, ...]
        student_w = np.concatenate((student_w, new_weight), axis=0)
        student_b = np.append(student_b, teacher_b[teacher_index])
    new_pre_layer = StubConv(layer.input_channel, n_pre_filters + n_add_filters, layer.kernel_size)
    new_pre_layer.set_weights((add_noise(student_w, teacher_w), add_noise(student_b, teacher_b)))
    return new_pre_layer


def wider_next_conv(layer, start_dim, total_dim, n_add, weighted=True):
    if not weighted:
        return StubConv(layer.input_channel + n_add, layer.filters, kernel_size=layer.kernel_size)
    n_filters = layer.filters
    teacher_w, teacher_b = layer.get_weights()

    new_weight_shape = list(teacher_w.shape)
    new_weight_shape[1] = n_add
    new_weight = np.zeros(tuple(new_weight_shape))

    student_w = np.concatenate((teacher_w[:, :start_dim, ...].copy(),
                                add_noise(new_weight, teacher_w),
                                teacher_w[:, start_dim:total_dim, ...].copy()), axis=1)
    new_layer = StubConv(layer.input_channel + n_add, n_filters, layer.kernel_size)
    new_layer.set_weights((student_w, teacher_b))
    return new_layer


def wider_bn(layer, start_dim, total_dim, n_add, weighted=True):
    if not weighted:
        return StubBatchNormalization(layer.num_features + n_add)

    weights = layer.get_weights()

    new_weights = [add_noise(np.ones(n_add, dtype=np.float32), np.array([0, 1])),
                   add_noise(np.zeros(n_add, dtype=np.float32), np.array([0, 1])),
                   add_noise(np.zeros(n_add, dtype=np.float32), np.array([0, 1])),
                   add_noise(np.ones(n_add, dtype=np.float32), np.array([0, 1]))]

    student_w = tuple()
    for weight, new_weight in zip(weights, new_weights):
        temp_w = weight.copy()
        temp_w = np.concatenate((temp_w[:start_dim], new_weight, temp_w[start_dim:total_dim]))
        student_w += (temp_w,)
    new_layer = StubBatchNormalization(layer.num_features + n_add)
    new_layer.set_weights(student_w)
    return new_layer


def wider_next_dense(layer, start_dim, total_dim, n_add, weighted=True):
    if not weighted:
        return StubDense(layer.input_units + n_add, layer.units)
    teacher_w, teacher_b = layer.get_weights()
    student_w = teacher_w.copy()
    n_units_each_channel = int(teacher_w.shape[1] / total_dim)

    new_weight = np.zeros((teacher_w.shape[0], n_add * n_units_each_channel))
    student_w = np.concatenate((student_w[:, :start_dim * n_units_each_channel],
                                add_noise(new_weight, student_w),
                                student_w[:, start_dim * n_units_each_channel:total_dim * n_units_each_channel]),
                               axis=1)

    new_layer = StubDense(layer.input_units + n_add, layer.units)
    new_layer.set_weights((student_w, teacher_b))
    return new_layer


def add_noise(weights, other_weights):
    w_range = np.ptp(other_weights.flatten())
    noise_range = NOISE_RATIO * w_range
    noise = np.random.uniform(-noise_range / 2.0, noise_range / 2.0, weights.shape)
    return np.add(noise, weights)
