import keras
import torch
from copy import deepcopy

from autokeras.nn.layers import set_stub_weight_to_torch, StubAdd, StubConcatenate, set_torch_weight_to_stub, \
    to_real_keras_layer, set_stub_weight_to_keras, set_keras_weight_to_stub


class TorchModel(torch.nn.Module):
    """A neural network class using pytorch constructed from an instance of Graph."""

    def __init__(self, graph):
        super(TorchModel, self).__init__()
        self.graph = graph
        self.layers = []
        for layer in graph.layer_list:
            self.layers.append(layer.to_real_layer())
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


class KerasModel:
    def __init__(self, graph):
        self.graph = graph
        self.layers = []
        for layer in graph.layer_list:
            self.layers.append(to_real_keras_layer(layer))

        # Construct the keras graph.
        # Input
        topo_node_list = self.graph.topological_order
        output_id = topo_node_list[-1]
        input_id = topo_node_list[0]
        input_tensor = keras.layers.Input(shape=graph.node_list[input_id].shape)

        node_list = deepcopy(self.graph.node_list)
        node_list[input_id] = input_tensor

        # Output
        for v in topo_node_list:
            for u, layer_id in self.graph.reverse_adj_list[v]:
                layer = self.graph.layer_list[layer_id]
                keras_layer = self.layers[layer_id]

                if isinstance(layer, (StubAdd, StubConcatenate)):
                    edge_input_tensor = list(map(lambda x: node_list[x],
                                                 self.graph.layer_id_to_input_node_ids[layer_id]))
                else:
                    edge_input_tensor = node_list[u]

                temp_tensor = keras_layer(edge_input_tensor)
                node_list[v] = temp_tensor

        output_tensor = node_list[output_id]
        self.model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)

        if graph.weighted:
            for index, layer in enumerate(self.layers):
                set_stub_weight_to_keras(self.graph.layer_list[index], layer)

    def set_weight_to_graph(self):
        self.graph.weighted = True
        for index, layer in enumerate(self.layers):
            set_keras_weight_to_stub(layer, self.graph.layer_list[index])