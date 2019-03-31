from autokeras.constant import Constant


class StubLayer:
    def __init__(self, input_node=None, output_node=None):
        self.input = input_node
        self.output = output_node
        self.weights = None  # (weight, bias) numpy.ndarray

    def build(self, shape):
        pass

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    @staticmethod
    def size():
        return 0

    @property
    def output_shape(self):
        return self.input.shape

    def __str__(self):
        return type(self).__name__[4:]


class StubBatchNormalization(StubLayer):

    def __init__(self, num_features, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.num_features = num_features

    def size(self):
        return self.num_features * 4


class StubBatchNormalization1d(StubBatchNormalization):
    pass


class StubBatchNormalization2d(StubBatchNormalization):
    pass


class StubBatchNormalization3d(StubBatchNormalization):
    pass


class StubDense(StubLayer):

    def __init__(self, input_units, units, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.input_units = input_units
        self.units = units

    @property
    def output_shape(self):
        return self.units,

    def size(self):
        return self.input_units * self.units + self.units


class StubConv(StubLayer):

    def __init__(self, input_channel, filters, kernel_size, stride=1, padding=None, output_node=None, input_node=None):
        super().__init__(input_node, output_node)
        self.input_channel = input_channel
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else int(self.kernel_size / 2)

    @property
    def output_shape(self):
        ret = list(self.input.shape[:-1])
        for index, dim in enumerate(ret):
            ret[index] = int((dim + 2 * self.padding - self.kernel_size) / self.stride) + 1
        ret = ret + [self.filters]
        return tuple(ret)

    def size(self):
        return (self.input_channel * self.kernel_size * self.kernel_size + 1) * self.filters

    def __str__(self):
        return super().__str__() + '(' + ', '.join(str(item) for item in [self.input_channel,
                                                                          self.filters,
                                                                          self.kernel_size,
                                                                          self.stride]) + ')'


class StubConv1d(StubConv):
    pass


class StubConv2d(StubConv):
    pass


class StubConv3d(StubConv):
    pass


class StubAggregateLayer(StubLayer):
    def __init__(self, input_nodes=None, output_node=None):
        if input_nodes is None:
            input_nodes = []
        super().__init__(input_nodes, output_node)


class StubConcatenate(StubAggregateLayer):

    @property
    def output_shape(self):
        ret = 0
        for current_input in self.input:
            ret += current_input.shape[-1]
        ret = self.input[0].shape[:-1] + (ret,)
        return ret


class StubAdd(StubAggregateLayer):

    @property
    def output_shape(self):
        return self.input[0].shape


class StubFlatten(StubLayer):

    @property
    def output_shape(self):
        ret = 1
        for dim in self.input.shape:
            ret *= dim
        return ret,


class StubReLU(StubLayer):
    pass


class StubSoftmax(StubLayer):
    pass


class StubPooling(StubLayer):

    def __init__(self,
                 kernel_size=None,
                 input_node=None,
                 output_node=None,
                 stride=None,
                 padding=0):
        super().__init__(input_node, output_node)
        self.kernel_size = kernel_size if kernel_size is not None else Constant.POOLING_KERNEL_SIZE
        self.stride = stride if stride is not None else self.kernel_size
        self.padding = padding

    @property
    def output_shape(self):
        ret = tuple()
        for dim in self.input.shape[:-1]:
            ret = ret + (max(int((dim + 2 * self.padding) / self.kernel_size), 1),)
        ret = ret + (self.input.shape[-1],)
        return ret


class StubPooling1d(StubPooling):
    pass


class StubPooling2d(StubPooling):
    pass


class StubPooling3d(StubPooling):
    pass


class StubAvgPooling1d(StubPooling):
    pass


class StubAvgPooling2d(StubPooling):
    pass


class StubAvgPooling3d(StubPooling):
    pass


class StubGlobalPooling(StubLayer):

    def __init__(self, input_node=None, output_node=None):
        super().__init__(input_node, output_node)

    @property
    def output_shape(self):
        return self.input.shape[-1],


class StubGlobalPooling1d(StubGlobalPooling):
    pass


class StubGlobalPooling2d(StubGlobalPooling):
    pass


class StubGlobalPooling3d(StubGlobalPooling):
    pass


class StubDropout(StubLayer):

    def __init__(self, rate, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.rate = rate


class StubDropout1d(StubDropout):
    pass


class StubDropout2d(StubDropout):
    pass


class StubDropout3d(StubDropout):
    pass


class StubInput(StubLayer):
    def __init__(self, input_node=None, output_node=None):
        super().__init__(input_node, output_node)


def layer_width(layer):
    if is_layer(layer, LayerType.DENSE):
        return layer.units
    if is_layer(layer, LayerType.CONV):
        return layer.filters
    print(layer)
    raise TypeError('The layer should be either Dense or Conv layer.')


def get_conv_class(n_dim):
    conv_class_list = [StubConv1d, StubConv2d, StubConv3d]
    return conv_class_list[n_dim - 1]


def get_dropout_class(n_dim):
    dropout_class_list = [StubDropout1d, StubDropout2d, StubDropout3d]
    return dropout_class_list[n_dim - 1]


def get_global_avg_pooling_class(n_dim):
    global_avg_pooling_class_list = [StubGlobalPooling1d, StubGlobalPooling2d, StubGlobalPooling3d]
    return global_avg_pooling_class_list[n_dim - 1]


def get_avg_pooling_class(n_dim):
    class_list = [StubAvgPooling1d, StubAvgPooling2d, StubAvgPooling3d]
    return class_list[n_dim - 1]


def get_pooling_class(n_dim):
    pooling_class_list = [StubPooling1d, StubPooling2d, StubPooling3d]
    return pooling_class_list[n_dim - 1]


def get_batch_norm_class(n_dim):
    batch_norm_class_list = [StubBatchNormalization1d, StubBatchNormalization2d, StubBatchNormalization3d]
    return batch_norm_class_list[n_dim - 1]


def get_n_dim(layer):
    if isinstance(layer, (StubConv1d, StubDropout1d, StubGlobalPooling1d, StubPooling1d, StubBatchNormalization1d)):
        return 1
    if isinstance(layer, (StubConv2d, StubDropout2d, StubGlobalPooling2d, StubPooling2d, StubBatchNormalization2d)):
        return 2
    if isinstance(layer, (StubConv3d, StubDropout3d, StubGlobalPooling3d, StubPooling3d, StubBatchNormalization3d)):
        return 3
    return -1


class LayerType:
    INPUT = (StubInput,)
    CONV = (StubConv,)
    DENSE = (StubDense,)
    BATCH_NORM = (StubBatchNormalization,)
    CONCAT = (StubConcatenate,)
    ADD = (StubAdd,)
    POOL = (StubPooling,)
    DROPOUT = (StubDropout,)
    SOFTMAX = (StubSoftmax,)
    RELU = (StubReLU,)
    FLATTEN = (StubFlatten,)
    GLOBAL_POOL = (StubGlobalPooling,)


def is_layer(layer, layer_type):
    return isinstance(layer, layer_type)
