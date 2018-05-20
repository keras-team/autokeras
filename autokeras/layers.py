from keras import backend
from keras.engine import Layer, InputLayer
from keras.layers import Add, Conv2D, Conv3D, Conv1D, Concatenate, Dense, BatchNormalization, Dropout, Activation, \
    Flatten, MaxPooling1D, MaxPooling2D, MaxPooling3D, GlobalAveragePooling1D, GlobalAveragePooling2D, \
    GlobalAveragePooling3D
from keras.regularizers import l2

from autokeras import constant


class WeightedAdd(Add):
    """Weighted Add class inherited from Add class

    It's used to do add weights for data in Add layer

    Attributes:
        weights: backend variable
        one: const 1.0
        kernel: None
        _trainable_weights: list that store weight
    """

    def __init__(self, **kwargs):
        """Init Weighted add class"""
        super(WeightedAdd, self).__init__(**kwargs)
        self.weight = backend.variable(1.0)
        self.one = backend.constant(1.0)
        self.kernel = None
        self._trainable_weights.append(self.weight)

    def call(self, x, **kwargs):
        """Override call function in Add and return new weights"""
        a = backend.tf.scalar_mul(self.weight, x[0])
        b = backend.tf.scalar_mul(backend.tf.subtract(self.one, self.weight), x[1])
        c = backend.tf.add(a, b)
        return c

    def compute_output_shape(self, input_shape):
        """Return output_shape"""
        return input_shape


class ConvConcat(Layer):
    def __init__(self, **kwargs):
        """Init Weighted add class"""
        super(ConvConcat, self).__init__(**kwargs)
        self.concatenate = None
        self.conv = None

    def build(self, input_shape):
        super(ConvConcat, self).build(input_shape)
        self.concatenate = Concatenate()
        self.conv = get_conv_layer_func(len(input_shape[0]) - 2)(input_shape[0][-1],
                                                                 kernel_size=1,
                                                                 padding='same',
                                                                 kernel_initializer='he_normal',
                                                                 kernel_regularizer=l2(1e-4))
        output_shape = self.concatenate.compute_output_shape(input_shape)
        self.conv.build(output_shape)

    def call(self, inputs, **kwargs):
        """Override call function in Add and return new weights"""
        super(ConvConcat, self).call(inputs)
        output = self.concatenate(inputs)
        output = self.conv(output)
        return output

    def compute_output_shape(self, input_shape):
        """Return output_shape"""
        return input_shape[0]

    def get_weights(self):
        return self.conv.get_weights()

    def set_weights(self, weights):
        self.conv.set_weights(weights)


class ConvBlock(Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        """Init Weighted add class"""
        super(ConvBlock, self).__init__(**kwargs)
        self.batch_normalization = None
        self.activation = None
        self.conv = None
        self.dropout = None
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        super(ConvBlock, self).build(input_shape)

        self.batch_normalization = BatchNormalization()
        self.batch_normalization.build(input_shape)
        output_shape = self.batch_normalization.compute_output_shape(input_shape)

        self.activation = Activation('relu')
        self.activation.build(output_shape)
        output_shape = self.activation.compute_output_shape(output_shape)

        self.conv = get_conv_layer_func(len(output_shape) - 2)(self.filters,
                                                               kernel_size=self.kernel_size,
                                                               padding='same',
                                                               kernel_initializer='he_normal',
                                                               kernel_regularizer=l2(1e-4))
        self.conv.build(output_shape)
        output_shape = self.conv.compute_output_shape(output_shape)

        self.dropout = Dropout(constant.CONV_DROPOUT_RATE)
        self.dropout.build(output_shape)

    def call(self, inputs, **kwargs):
        """Override call function in Add and return new weights"""
        super(ConvBlock, self).call(inputs)
        output = self.batch_normalization(inputs)
        output = self.activation(output)
        output = self.conv(output)
        output = self.dropout(output)
        return output

    def compute_output_shape(self, input_shape):
        """Return output_shape"""
        return input_shape[:-1] + (self.filters,)

    def get_weights(self):
        return [self.batch_normalization.get_weights(), self.conv.get_weights()]

    def set_weights(self, weights):
        self.batch_normalization.set_weights(weights[0])
        self.conv.set_weights(weights[1])

    def get_config(self):
        return {'filters': self.filters, 'kernel_size': self.kernel_size}


class StubLayer:
    def __init__(self, input_node=None, output_node=None):
        self.input = input_node
        self.output = output_node
        self.weights = None
        self.input_shape = None
        self.output_shape = None

    def build(self, shape):
        pass

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights


class StubBatchNormalization(StubLayer):
    pass


class StubDense(StubLayer):
    def __init__(self, units, activation, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.units = units
        self.output_shape = (None, units)
        self.activation = activation


class StubConv(StubLayer):
    def __init__(self, filters, kernel_size, func, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.filters = filters
        self.output_shape = (None, filters)
        self.kernel_size = kernel_size
        self.func = func
        if func is Conv1D:
            self.n_dim = 1
        if func is Conv2D:
            self.n_dim = 2
        if func is Conv3D:
            self.n_dim = 3


class StubConvBlock(StubLayer):
    def __init__(self, filters, kernel_size, add_skip, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.filters = filters
        self.kernel_size = kernel_size
        self.add_skip = add_skip


class StubAggregateLayer(StubLayer):
    def __init__(self, input_nodes=None, output_node=None):
        if input_nodes is None:
            input_nodes = []
        super().__init__(input_nodes, output_node)


class StubConvConcat(StubAggregateLayer):
    pass


class StubConcatenate(StubAggregateLayer):
    pass


class StubWeightedAdd(StubAggregateLayer):
    pass


class StubFlatten(StubLayer):
    pass


class StubActivation(StubLayer):
    def __init__(self, func, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.func = func


class StubPooling(StubLayer):
    def __init__(self, func, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.func = func


class StubGlobalPooling(StubLayer):
    def __init__(self, func, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.func = func


class StubDropout(StubLayer):
    def __init__(self, rate, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.rate = rate


class StubInput(StubLayer):
    def __init__(self, units, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.units = units
        self.output_shape = (None, units)


def is_layer(layer, layer_type):
    if layer_type == 'InputLayer':
        return isinstance(layer, InputLayer)
    if layer_type == 'Conv':
        return isinstance(layer, StubConv) or is_conv_layer(layer)
    if layer_type == 'Dense':
        return isinstance(layer, (StubDense, Dense))
    if layer_type == 'BatchNormalization':
        return isinstance(layer, (StubBatchNormalization, BatchNormalization))
    if layer_type == 'Concatenate':
        return isinstance(layer, (StubConcatenate, Concatenate))
    if layer_type == 'WeightedAdd':
        return isinstance(layer, (StubWeightedAdd, WeightedAdd))
    if layer_type == 'Pooling':
        return isinstance(layer, StubPooling) or is_pooling_layer(layer)
    if layer_type == 'Dropout':
        return isinstance(layer, (StubDropout, Dropout))
    if layer_type == 'Activation':
        return isinstance(layer, (StubActivation, Activation))
    if layer_type == 'Flatten':
        return isinstance(layer, (StubFlatten, Flatten))
    if layer_type == 'GlobalAveragePooling':
        return isinstance(layer, StubGlobalPooling) or is_global_pooling_layer(layer)


def layer_width(layer):
    if is_layer(layer, 'Dense'):
        return layer.units
    if is_layer(layer, 'Conv'):
        return layer.filters
    raise TypeError('The layer should be either Dense or Conv layer.')


def is_pooling_layer(layer):
    return isinstance(layer, (MaxPooling1D, MaxPooling2D, MaxPooling3D))


def is_global_pooling_layer(layer):
    return isinstance(layer, (GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D))


def is_conv_layer(layer):
    """Return whether the layer is convolution layer"""
    return isinstance(layer, tuple(constant.CONV_FUNC_LIST))


def is_dense_layer(layer):
    return isinstance(layer, Dense)


def get_conv_layer_func(n_dim):
    """Return convolution function based on the dimension"""
    conv_layer_functions = [Conv1D, Conv2D, Conv3D]
    if n_dim > 3:
        raise ValueError('The input dimension is too high.')
    if n_dim < 1:
        raise ValueError('The input dimension is too low.')
    return conv_layer_functions[n_dim - 1]


def get_ave_layer_func(n_dim):
    """Return convolution function based on the dimension"""
    conv_layer_functions = [GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D]
    if n_dim > 3:
        raise ValueError('The input dimension is too high.')
    if n_dim < 1:
        raise ValueError('The input dimension is too low.')
    return conv_layer_functions[n_dim - 1]


def to_stub_layer(input_id, layer, output_id):
    if is_layer(layer, 'Conv'):
        temp_stub_layer = StubConv(layer.filters, layer.kernel_size, layer.__class__, input_id, output_id)
    elif is_layer(layer, 'Dense'):
        temp_stub_layer = StubDense(layer.units, layer.activation, input_id, output_id)
    elif is_layer(layer, 'WeightedAdd'):
        temp_stub_layer = StubWeightedAdd(input_id, output_id)
    elif is_layer(layer, 'Concatenate'):
        temp_stub_layer = StubConcatenate(input_id, output_id)
    elif is_layer(layer, 'BatchNormalization'):
        temp_stub_layer = StubBatchNormalization(input_id, output_id)
    elif is_layer(layer, 'Activation'):
        temp_stub_layer = StubActivation(layer.activation, input_id, output_id)
    elif is_layer(layer, 'InputLayer'):
        temp_stub_layer = StubLayer(input_id, output_id)
    elif is_layer(layer, 'Flatten'):
        temp_stub_layer = StubFlatten(input_id, output_id)
    elif is_layer(layer, 'Dropout'):
        temp_stub_layer = StubDropout(layer.rate, input_id, output_id)
    elif is_layer(layer, 'Pooling'):
        temp_stub_layer = StubPooling(layer.__class__, input_id, output_id)
    elif is_layer(layer, 'GloabalAveragePooling'):
        temp_stub_layer = StubGlobalPooling(layer.__class__, input_id, output_id)
    elif is_layer(layer, 'ConvBlock'):
        temp_stub_layer = StubConvBlock(layer.filters, layer.kernel_size, input_id, output_id)
    elif is_layer(layer, 'ConvConcat'):
        temp_stub_layer = StubConvConcat(input_id, output_id)
    else:
        raise TypeError("The layer {} is illegal.".format(layer))
    return temp_stub_layer


def to_real_layer(layer):
    if is_layer(layer, 'Dense'):
        return Dense(layer.units, activation=layer.activation)
    if is_layer(layer, 'Conv'):
        return layer.func(layer.filters,
                          kernel_size=layer.kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))
    if is_layer(layer, 'Pooling'):
        return layer.func(padding='same')
    if is_layer(layer, 'BatchNormalization'):
        return BatchNormalization()
    if is_layer(layer, 'Concatenate'):
        return Concatenate()
    if is_layer(layer, 'WeightedAdd'):
        return WeightedAdd()
    if is_layer(layer, 'Dropout'):
        return Dropout(layer.rate)
    if is_layer(layer, 'Activation'):
        return Activation(layer.func)
    if is_layer(layer, 'Flatten'):
        return Flatten()
    if is_layer(layer, 'GlobalAveragePooling'):
        return layer.func()