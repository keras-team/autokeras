from queue import Queue

from keras import Input
from keras.engine import Model
from keras.layers import BatchNormalization

from autokeras.layer_transformer import deeper_conv_block, conv_to_wider_layer
from autokeras.utils import copy_layer, is_conv_layer, get_int_tuple


class Graph:
    def __init__(self, model):
        layers = model.layers[1:]
        self.model = model
        self.node_list = []
        self.edge_list = []
        # node id start with 0
        self.node_to_id = {}
        self.edge_to_id = {}
        self.old_edge_ids = {}
        self.adj_list = {}

        # Add all nodes
        for layer in layers:
            self._add_node(layer.input)
            self._add_node(layer.output)

        # Add all edges
        for layer in layers:
            self._add_edge(layer)

    @property
    def n_nodes(self):
        return len(self.node_list)

    @property
    def n_edges(self):
        return len(self.edge_list)

    def _add_node(self, node):
        if node not in self.node_list:
            current_id = len(self.node_list)
            self.node_to_id[node] = current_id
            self.node_list.append(node)
            self.adj_list[current_id] = []

    def _add_edge(self, layer, input_id=None, output_id=None, old=True):
        if input_id is None:
            input_id = self.node_to_id[layer.input]
        if output_id is None:
            output_id = self.node_to_id[layer.output]
        current_id = len(self.edge_list)
        self.adj_list[input_id].append((output_id, current_id))
        self.edge_list.append(layer)
        self.edge_to_id[layer] = current_id
        if old:
            self.old_edge_ids[current_id] = True

    def to_deeper_model(self, target, kernel_size):
        new_layers = deeper_conv_block(target, kernel_size)
        output_id = self.node_to_id[target.output]
        output_id = self.adj_list[output_id][0][0]

        for i in range(3):
            self._add_node(i)

        self._add_edge(new_layers[0], self.node_to_id[0], self.node_to_id[1], False)
        self._add_edge(new_layers[1], self.node_to_id[1], self.node_to_id[2], False)
        self._add_edge(new_layers[2], self.node_to_id[2], self.adj_list[output_id][0][0], False)
        original_relu_edge = self.adj_list[output_id][0]
        self.adj_list[output_id][0] = (self.node_to_id[0], original_relu_edge[1])

        return self.produce_model()

    def produce_model(self):
        input_shape = get_int_tuple(self.model.inputs[0].shape[1:])

        input_tensor = Input(shape=tuple(input_shape))
        input_id = self.node_to_id[self.model.inputs[0]]
        output_id = self.node_to_id[self.model.outputs[0]]

        id_to_tensor = {input_id: input_tensor}
        q = Queue()
        q.put(input_id)
        while not q.empty():
            u = q.get()
            for v, edge_id in self.adj_list[u]:
                layer = self.edge_list[edge_id]
                if edge_id in self.old_edge_ids:
                    copied_layer = copy_layer(layer, get_int_tuple(id_to_tensor[u].shape))
                    temp_tensor = copied_layer(id_to_tensor[u])
                else:
                    temp_tensor = layer(id_to_tensor[u])
                id_to_tensor[v] = temp_tensor
                q.put(v)
        return Model(input_tensor, id_to_tensor[output_id])

    def to_wider_model(self, pre_layer, n_add):
        output_id = self.node_to_id[pre_layer.output]
        next_conv_layer_list, next_bn_layer_list = self._search_following_conv(output_id)
        new_pre_layer, new_next_layer_list, new_bn_layer_list = conv_to_wider_layer(pre_layer,
                                                                                    next_conv_layer_list,
                                                                                    next_bn_layer_list,
                                                                                    n_add)
        for old_layer, new_layer in zip(next_conv_layer_list, new_next_layer_list):
            self._replace_edge(old_layer, new_layer)
        for old_layer, new_layer in zip(next_bn_layer_list, new_bn_layer_list):
            self._replace_edge(old_layer, new_layer)
        self._replace_edge(pre_layer, new_pre_layer)
        return self.produce_model()

    def _search_following_conv(self, node_id):
        stack = [node_id]
        conv_list = []
        bn_list = []
        while stack:
            u = stack.pop()
            for v, edge_id in self.adj_list[u]:
                layer = self.edge_list[edge_id]
                if is_conv_layer(layer):
                    conv_list.append(layer)
                elif isinstance(layer, BatchNormalization):
                    bn_list.append(layer)
                    stack.append(v)
                else:
                    stack.append(v)
        return conv_list, bn_list

    def _replace_edge(self, old_layer, new_layer):
        edge_id = self.edge_to_id[old_layer]
        self.edge_list[edge_id] = new_layer
        self.edge_to_id[new_layer] = edge_id
        self.edge_to_id.pop(old_layer)
        self.old_edge_ids.pop(edge_id)
