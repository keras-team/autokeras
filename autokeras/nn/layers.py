from abc import abstractmethod

import torch
from torch import nn
from torch.nn import functional

from autokeras.constant import Constant


class TorchConcatenate(nn.Module):
    @staticmethod
    def forward(input_list):
        return torch.cat(input_list, dim=1)


class TorchAdd(nn.Module):
    @staticmethod
    def forward(input_list):
        return input_list[0] + input_list[1]


class TorchFlatten(nn.Module):
    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, input_tensor):
        pass


class GlobalAvgPool1d(AvgPool):
    def forward(self, input_tensor):
        return functional.avg_pool1d(input_tensor, input_tensor.size()[2:]).view(input_tensor.size()[:2])


class GlobalAvgPool2d(AvgPool):
    def forward(self, input_tensor):
        return functional.avg_pool2d(input_tensor, input_tensor.size()[2:]).view(input_tensor.size()[:2])


class GlobalAvgPool3d(AvgPool):
    def forward(self, input_tensor):
        return functional.avg_pool3d(input_tensor, input_tensor.size()[2:]).view(input_tensor.size()[:2])


class StubLayer:

    def __init__(self, input_node=None, output_node=None):
        self.input = input_node
        self.output = output_node
        self.weights = None

    def build(self, shape):
        pass

    def set_weights(self, weights):
        self.weights = weights

    def import_weights(self, torch_layer):
        pass

    def export_weights(self, torch_layer):
        pass

    def get_weights(self):
        return self.weights

    @staticmethod
    def size():
        return 0

    @property
    def output_shape(self):
        return self.input.shape

    def to_real_layer(self):
        pass

    def __str__(self):
        return type(self).__name__[4:]


class StubWeightBiasLayer(StubLayer):

    def import_weights(self, torch_layer):
        self.set_weights((torch_layer.weight.data.cpu().numpy(), torch_layer.bias.data.cpu().numpy()))

    def export_weights(self, torch_layer):
        torch_layer.weight.data = torch.Tensor(self.weights[0])
        torch_layer.bias.data = torch.Tensor(self.weights[1])


class StubBatchNormalization(StubWeightBiasLayer):

    def __init__(self, num_features, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.num_features = num_features

    def import_weights(self, torch_layer):
        self.set_weights((torch_layer.weight.data.cpu().numpy(),
                          torch_layer.bias.data.cpu().numpy(),
                          torch_layer.running_mean.cpu().numpy(),
                          torch_layer.running_var.cpu().numpy(),
                          ))

    def export_weights(self, torch_layer):
        torch_layer.weight.data = torch.Tensor(self.weights[0])
        torch_layer.bias.data = torch.Tensor(self.weights[1])
        torch_layer.running_mean = torch.Tensor(self.weights[2])
        torch_layer.running_var = torch.Tensor(self.weights[3])

    def size(self):
        return self.num_features * 4

    @abstractmethod
    def to_real_layer(self):
        pass


class StubBatchNormalization1d(StubBatchNormalization):
    def to_real_layer(self):
        return torch.nn.BatchNorm1d(self.num_features)


class StubBatchNormalization2d(StubBatchNormalization):
    def to_real_layer(self):
        return torch.nn.BatchNorm2d(self.num_features)


class StubBatchNormalization3d(StubBatchNormalization):
    def to_real_layer(self):
        return torch.nn.BatchNorm3d(self.num_features)


class StubDense(StubWeightBiasLayer):

    def __init__(self, input_units, units, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.input_units = input_units
        self.units = units

    @property
    def output_shape(self):
        return self.units,

    def size(self):
        return self.input_units * self.units + self.units

    def to_real_layer(self):
        return torch.nn.Linear(self.input_units, self.units)


class StubConv(StubWeightBiasLayer):

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
        return self.filters * self.kernel_size * self.kernel_size + self.filters

    @abstractmethod
    def to_real_layer(self):
        pass

    def __str__(self):
        return super().__str__() + '(' + ', '.join(str(item) for item in [self.input_channel,
                                                                          self.filters,
                                                                          self.kernel_size,
                                                                          self.stride]) + ')'


class StubConv1d(StubConv):
    def to_real_layer(self):
        return torch.nn.Conv1d(self.input_channel,
                               self.filters,
                               self.kernel_size,
                               stride=self.stride,
                               padding=self.padding)


class StubConv2d(StubConv):
    def to_real_layer(self):
        return torch.nn.Conv2d(self.input_channel,
                               self.filters,
                               self.kernel_size,
                               stride=self.stride,
                               padding=self.padding)


class StubConv3d(StubConv):
    def to_real_layer(self):
        return torch.nn.Conv3d(self.input_channel,
                               self.filters,
                               self.kernel_size,
                               stride=self.stride,
                               padding=self.padding)


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

    def to_real_layer(self):
        return TorchConcatenate()


class StubAdd(StubAggregateLayer):

    @property
    def output_shape(self):
        return self.input[0].shape

    def to_real_layer(self):
        return TorchAdd()


class StubFlatten(StubLayer):

    @property
    def output_shape(self):
        ret = 1
        for dim in self.input.shape:
            ret *= dim
        return ret,

    def to_real_layer(self):
        return TorchFlatten()


class StubReLU(StubLayer):
    def to_real_layer(self):
        return torch.nn.ReLU()


class StubSoftmax(StubLayer):
    def to_real_layer(self):
        return torch.nn.LogSoftmax(dim=1)


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

    @abstractmethod
    def to_real_layer(self):
        pass


class StubPooling1d(StubPooling):
    def to_real_layer(self):
        return torch.nn.MaxPool1d(self.kernel_size, stride=self.stride)


class StubPooling2d(StubPooling):
    def to_real_layer(self):
        return torch.nn.MaxPool2d(self.kernel_size, stride=self.stride)


class StubPooling3d(StubPooling):
    def to_real_layer(self):
        return torch.nn.MaxPool3d(self.kernel_size, stride=self.stride)


class StubAvgPooling1d(StubPooling):
    def to_real_layer(self):
        return torch.nn.AvgPool1d(self.kernel_size, stride=self.stride)


class StubAvgPooling2d(StubPooling):
    def to_real_layer(self):
        return torch.nn.AvgPool2d(self.kernel_size, stride=self.stride)


class StubAvgPooling3d(StubPooling):
    def to_real_layer(self):
        return torch.nn.AvgPool3d(self.kernel_size, stride=self.stride)


class StubGlobalPooling(StubLayer):

    def __init__(self, input_node=None, output_node=None):
        super().__init__(input_node, output_node)

    @property
    def output_shape(self):
        return self.input.shape[-1],

    @abstractmethod
    def to_real_layer(self):
        pass


class StubGlobalPooling1d(StubGlobalPooling):
    def to_real_layer(self):
        return GlobalAvgPool1d()


class StubGlobalPooling2d(StubGlobalPooling):
    def to_real_layer(self):
        return GlobalAvgPool2d()


class StubGlobalPooling3d(StubGlobalPooling):
    def to_real_layer(self):
        return GlobalAvgPool3d()


class StubDropout(StubLayer):

    def __init__(self, rate, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.rate = rate

    @abstractmethod
    def to_real_layer(self):
        pass


class StubDropout1d(StubDropout):
    def to_real_layer(self):
        return torch.nn.Dropout(self.rate)


class StubDropout2d(StubDropout):
    def to_real_layer(self):
        return torch.nn.Dropout2d(self.rate)


class StubDropout3d(StubDropout):
    def to_real_layer(self):
        return torch.nn.Dropout3d(self.rate)


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


def set_torch_weight_to_stub(torch_layer, stub_layer):
    stub_layer.import_weights(torch_layer)


def set_stub_weight_to_torch(stub_layer, torch_layer):
    stub_layer.export_weights(torch_layer)


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
