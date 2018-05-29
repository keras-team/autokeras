from keras import backend
from keras.engine import InputLayer
from keras.layers import Add, Conv2D, Conv3D, Conv1D, Dense, BatchNormalization, Concatenate, Dropout, Activation, \
    Flatten, MaxPooling1D, MaxPooling2D, MaxPooling3D, GlobalAveragePooling1D, GlobalAveragePooling2D, \
    GlobalAveragePooling3D
from keras.regularizers import l2


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


class StubAggregateLayer(StubLayer):
    def __init__(self, input_nodes=None, output_node=None):
        if input_nodes is None:
            input_nodes = []
        super().__init__(input_nodes, output_node)


class StubConcatenate(StubAggregateLayer):
    pass


class StubWeightedAdd(StubAggregateLayer):
    pass


class StubFlatten(StubLayer):
    pass


class StubActivation(StubLayer):
    def __init__(self, activation, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.activation = activation


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
    def __init__(self, input_node=None, output_node=None):
        super().__init__(input_node, output_node)


def is_layer(layer, layer_type):
    if layer_type == 'Input':
        return isinstance(layer, (InputLayer, StubInput))
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


def is_conv_layer(layer):
    """Return whether the layer is convolution layer"""
    return isinstance(layer, tuple(CONV_FUNC_LIST))


def is_dense_layer(layer):
    return isinstance(layer, Dense)


def to_real_layer(layer):
    if is_layer(layer, 'Dense'):
        return Dense(layer.units, activation=layer.activation)
    if is_layer(layer, 'Conv'):
        return Conv2D(layer.filters,
                      kernel_size=layer.kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))
    if is_layer(layer, 'Pooling'):
        return MaxPooling2D(padding='same')
    if is_layer(layer, 'BatchNormalization'):
        return BatchNormalization()
    if is_layer(layer, 'Concatenate'):
        return Concatenate()
    if is_layer(layer, 'WeightedAdd'):
        return WeightedAdd()
    if is_layer(layer, 'Dropout'):
        return Dropout(layer.rate)
    if is_layer(layer, 'Activation'):
        return Activation(layer.activation)
    if is_layer(layer, 'Flatten'):
        return Flatten()
    if is_layer(layer, 'GlobalAveragePooling'):
        return GlobalAveragePooling2D()


def to_stub_layer(layer, input_id, output_id):
    if is_conv_layer(layer):
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
    elif is_layer(layer, 'Input'):
        temp_stub_layer = StubInput(input_id, output_id)
    elif is_layer(layer, 'Flatten'):
        temp_stub_layer = StubFlatten(input_id, output_id)
    elif is_layer(layer, 'Dropout'):
        temp_stub_layer = StubDropout(layer.rate, input_id, output_id)
    elif is_layer(layer, 'Pooling'):
        temp_stub_layer = StubPooling(layer.__class__, input_id, output_id)
    elif is_layer(layer, 'GlobalAveragePooling'):
        temp_stub_layer = StubGlobalPooling(layer.__class__, input_id, output_id)
    else:
        raise TypeError("The layer {} is illegal.".format(layer))
    return temp_stub_layer


CONV_FUNC_LIST = [Conv1D, Conv2D, Conv3D, StubConv]
WEIGHTED_LAYER_FUNC_LIST = CONV_FUNC_LIST + [Dense, StubDense]
