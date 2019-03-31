import torch
from abc import abstractmethod
from copy import deepcopy
from torch import nn
from torch.nn import functional

from autokeras.nn.layers import StubAdd, StubConcatenate, StubConv1d, StubConv2d, StubConv3d, \
    StubDropout1d, StubDropout2d, StubDropout3d, StubGlobalPooling1d, StubGlobalPooling2d, StubGlobalPooling3d, \
    StubPooling1d, StubPooling2d, StubPooling3d, StubBatchNormalization1d, StubBatchNormalization2d, \
    StubBatchNormalization3d, StubSoftmax, StubReLU, StubFlatten, StubDense, StubAvgPooling1d, StubAvgPooling2d, \
    StubAvgPooling3d


def produce_model(graph):
    return TorchModel(graph)


class TorchModel(torch.nn.Module):
    """A neural network class using pytorch constructed from an instance of Graph."""

    def __init__(self, graph):
        super(TorchModel, self).__init__()
        self.graph = graph
        self.layers = []
        for layer in graph.layer_list:
            self.layers.append(to_real_layer(layer))
        if graph.weighted:
            for index, layer in enumerate(self.layers):
                set_stub_weight_to_torch(self.graph.layer_list[index], layer)
        for index, layer in enumerate(self.layers):
            self.add_module(str(index), layer)

    def forward(self, input_tensor):
        topo_node_list = self.graph.topological_order
        output_id = topo_node_list[-1]
        input_id = topo_node_list[0]

        node_list = deepcopy(self.graph.node_list)
        node_list[input_id] = input_tensor

        for v in topo_node_list:
            for u, layer_id in self.graph.reverse_adj_list[v]:
                layer = self.graph.layer_list[layer_id]
                torch_layer = list(self.modules())[layer_id + 1]

                if isinstance(layer, (StubAdd, StubConcatenate)):
                    edge_input_tensor = list(map(lambda x: node_list[x],
                                                 self.graph.layer_id_to_input_node_ids[layer_id]))
                else:
                    edge_input_tensor = node_list[u]
                temp_tensor = torch_layer(edge_input_tensor)
                node_list[v] = temp_tensor
        return node_list[output_id]

    def set_weight_to_graph(self):
        self.graph.weighted = True
        for index, layer in enumerate(self.layers):
            set_torch_weight_to_stub(layer, self.graph.layer_list[index])


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


def set_torch_weight_to_stub(torch_layer, stub_layer):
    # stub_layer.import_weights(torch_layer)
    if isinstance(stub_layer, (StubConv1d, StubConv2d, StubConv3d, StubDense)):
        stub_layer.set_weights((torch_layer.weight.data.cpu().numpy(), torch_layer.bias.data.cpu().numpy()))
    elif isinstance(stub_layer, (StubBatchNormalization1d, StubBatchNormalization2d, StubBatchNormalization3d)):
        stub_layer.set_weights((torch_layer.weight.data.cpu().numpy(),
                          torch_layer.bias.data.cpu().numpy(),
                          torch_layer.running_mean.cpu().numpy(),
                          torch_layer.running_var.cpu().numpy(),
                          ))


def set_stub_weight_to_torch(stub_layer, torch_layer):
    # stub_layer.export_weights(torch_layer)
    if isinstance(stub_layer, (StubConv1d, StubConv2d, StubConv3d, StubDense)):
        torch_layer.weight.data = torch.Tensor(stub_layer.weights[0])
        torch_layer.bias.data = torch.Tensor(stub_layer.weights[1])
    elif isinstance(stub_layer, (StubBatchNormalization1d, StubBatchNormalization2d, StubBatchNormalization3d)):
        torch_layer.weight.data = torch.Tensor(stub_layer.weights[0])
        torch_layer.bias.data = torch.Tensor(stub_layer.weights[1])
        torch_layer.running_mean = torch.Tensor(stub_layer.weights[2])
        torch_layer.running_var = torch.Tensor(stub_layer.weights[3])


def to_real_layer(stub_layer):
    if isinstance(stub_layer, StubConv1d):
        return torch.nn.Conv1d(stub_layer.input_channel,
                               stub_layer.filters,
                               stub_layer.kernel_size,
                               stride=stub_layer.stride,
                               padding=stub_layer.padding)

    elif isinstance(stub_layer, StubConv2d):
        return torch.nn.Conv2d(stub_layer.input_channel,
                               stub_layer.filters,
                               stub_layer.kernel_size,
                               stride=stub_layer.stride,
                               padding=stub_layer.padding)

    elif isinstance(stub_layer, StubConv3d):
        return torch.nn.Conv3d(stub_layer.input_channel,
                               stub_layer.filters,
                               stub_layer.kernel_size,
                               stride=stub_layer.stride,
                               padding=stub_layer.padding)

    elif isinstance(stub_layer, StubDropout1d):
        return torch.nn.Dropout(stub_layer.rate)
    elif isinstance(stub_layer, StubDropout2d):
        return torch.nn.Dropout2d(stub_layer.rate)
    elif isinstance(stub_layer, StubDropout3d):
        return torch.nn.Dropout3d(stub_layer.rate)
    elif isinstance(stub_layer, StubAvgPooling1d):
        return torch.nn.AvgPool1d(stub_layer.kernel_size, stride=stub_layer.stride)
    elif isinstance(stub_layer, StubAvgPooling2d):
        return torch.nn.AvgPool2d(stub_layer.kernel_size, stride=stub_layer.stride)
    elif isinstance(stub_layer, StubAvgPooling3d):
        return torch.nn.AvgPool3d(stub_layer.kernel_size, stride=stub_layer.stride)
    elif isinstance(stub_layer, StubGlobalPooling1d):
        return GlobalAvgPool1d()
    elif isinstance(stub_layer, StubGlobalPooling2d):
        return GlobalAvgPool2d()
    elif isinstance(stub_layer, StubGlobalPooling3d):
        return GlobalAvgPool3d()
    elif isinstance(stub_layer, StubPooling1d):
        return torch.nn.MaxPool1d(stub_layer.kernel_size, stride=stub_layer.stride)
    elif isinstance(stub_layer, StubPooling2d):
        return torch.nn.MaxPool2d(stub_layer.kernel_size, stride=stub_layer.stride)
    elif isinstance(stub_layer, StubPooling3d):
        return torch.nn.MaxPool3d(stub_layer.kernel_size, stride=stub_layer.stride)
    elif isinstance(stub_layer, StubBatchNormalization1d):
        return torch.nn.BatchNorm1d(stub_layer.num_features)
    elif isinstance(stub_layer, StubBatchNormalization2d):
        return torch.nn.BatchNorm2d(stub_layer.num_features)
    elif isinstance(stub_layer, StubBatchNormalization3d):
        return torch.nn.BatchNorm3d(stub_layer.num_features)
    elif isinstance(stub_layer, StubSoftmax):
        return torch.nn.LogSoftmax(dim=1)
    elif isinstance(stub_layer, StubReLU):
        return torch.nn.ReLU()
    elif isinstance(stub_layer, StubFlatten):
        return TorchFlatten()
    elif isinstance(stub_layer, StubAdd):
        return TorchAdd()
    elif isinstance(stub_layer, StubConcatenate):
        return TorchConcatenate()
    elif isinstance(stub_layer, StubDense):
        return torch.nn.Linear(stub_layer.input_units, stub_layer.units)
