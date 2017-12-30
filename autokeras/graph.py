from queue import Queue

from keras import Input
from keras.engine import Model
from keras.layers import BatchNormalization, Dense

from autokeras.layer_transformer import deeper_conv_block, conv_to_wider_layer, dense_to_wider_layer
from autokeras.layers import WeightedAdd
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
        self.edge_id_to_input_ids = {}
        self.old_edge_ids = {}
        self.adj_list = {}
        self.reverse_adj_list = {}

        # Add all nodes
        for layer in layers:
            if isinstance(layer.input, list):
                for temp_input in layer.input:
                    self._add_node(temp_input)
            else:
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
            node_id = len(self.node_list)
            self.node_to_id[node] = node_id
            self.node_list.append(node)
            self.adj_list[node_id] = []
            self.reverse_adj_list[node_id] = []

    def _add_edge(self, layer, input_id=None, output_id=None, old=True):
        if input_id is None:
            if isinstance(layer.input, list):
                for temp_input in layer.input:
                    input_id = self.node_to_id[temp_input]
                    self._add_edge(layer, input_id, output_id, old)
                return
            input_id = self.node_to_id[layer.input]

        if output_id is None:
            output_id = self.node_to_id[layer.output]

        if layer in self.edge_to_id:
            edge_id = self.edge_to_id[layer]
            self.edge_id_to_input_ids[edge_id].append(input_id)
        else:
            edge_id = len(self.edge_list)
            self.edge_list.append(layer)
            self.edge_to_id[layer] = edge_id
            self.edge_id_to_input_ids[edge_id] = [input_id]

        self.adj_list[input_id].append((output_id, edge_id))
        self.reverse_adj_list[output_id].append((input_id, edge_id))

        if old:
            self.old_edge_ids[edge_id] = True

    def to_add_skip_model(self, start, end):
        input_id = self.node_to_id[start.input]
        output_id = self.node_to_id[end.output]
        output_id = self.adj_list[output_id][0][0]

        self._add_node(0)
        new_node_id = self.node_to_id[0]
        layer = WeightedAdd()
        single_input_shape = get_int_tuple(self.adj_list[output_id][0][1].input_shape)
        layer.build([single_input_shape, single_input_shape])
        self._add_edge(layer, new_node_id, self.adj_list[output_id][0][0], False)
        self._add_edge(layer, input_id, self.adj_list[output_id][0][0], False)

        self._redirect_edge(output_id, self.adj_list[output_id][0][0], new_node_id)

    def _redirect_edge(self, u_id, v_id, new_v_id):
        layer_id = None
        for index, edge_tuple in enumerate(self.adj_list[u_id]):
            if edge_tuple[0] == v_id:
                layer_id = edge_tuple[1]
                self.adj_list[u_id].remove(edge_tuple)
                break

        for index, edge_tuple in enumerate(self.reverse_adj_list[v_id]):
            if edge_tuple[0] == u_id:
                layer_id = edge_tuple[1]
                self.reverse_adj_list[v_id].remove(edge_tuple)
                break

        self._add_edge(self.edge_list[layer_id], u_id, new_v_id)

    def to_conv_deeper_model(self, target, kernel_size):
        new_layers = deeper_conv_block(target, kernel_size)
        output_id = self.node_to_id[target.output]
        output_id = self.adj_list[output_id][0][0]

        for i in range(3):
            self._add_node(i)

        self._add_edge(new_layers[0], self.node_to_id[0], self.node_to_id[1], False)
        self._add_edge(new_layers[1], self.node_to_id[1], self.node_to_id[2], False)
        self._add_edge(new_layers[2], self.node_to_id[2], self.adj_list[output_id][0][0], False)
        self._redirect_edge(output_id, self.adj_list[output_id][0][0], self.node_to_id[0])

        return self.produce_model()

    def produce_model(self):
        input_tensor = Input(shape=get_int_tuple(self.model.inputs[0].shape[1:]))
        input_id = self.node_to_id[self.model.inputs[0]]
        output_id = self.node_to_id[self.model.outputs[0]]

        id_to_tensor = {input_id: input_tensor}
        for v in self._topological_order():
            for u, edge_id in self.reverse_adj_list[v]:
                layer = self.edge_list[edge_id]

                if isinstance(layer, WeightedAdd):
                    edge_input_tensor = list(map(lambda x: id_to_tensor[x], self.edge_id_to_input_ids[edge_id]))
                else:
                    edge_input_tensor = id_to_tensor[u]

                if edge_id in self.old_edge_ids:
                    new_layer = copy_layer(layer)
                else:
                    new_layer = layer

                temp_tensor = new_layer(edge_input_tensor)
                id_to_tensor[v] = temp_tensor
        return Model(input_tensor, id_to_tensor[output_id])

    def to_conv_wider_model(self, pre_layer, n_add):
        output_id = self.node_to_id[pre_layer.output]
        next_layer_list = self._search_following_conv(output_id)
        new_pre_layer, new_next_layer_list = conv_to_wider_layer(pre_layer,
                                                                 next_layer_list,
                                                                 n_add)
        for old_layer, new_layer in zip(next_layer_list, new_next_layer_list):
            self._replace_edge(old_layer, new_layer)
        self._replace_edge(pre_layer, new_pre_layer)
        return self.produce_model()

    def _search_following_conv(self, node_id):
        stack = [node_id]
        ret_list = []
        while stack:
            u = stack.pop()
            for v, edge_id in self.adj_list[u]:
                layer = self.edge_list[edge_id]
                if is_conv_layer(layer):
                    ret_list.append(layer)
                elif isinstance(layer, BatchNormalization):
                    ret_list.append(layer)
                    stack.append(v)
                elif isinstance(layer, Dense):
                    ret_list.append(layer)
                else:
                    stack.append(v)
        return ret_list

    def _replace_edge(self, old_layer, new_layer):
        edge_id = self.edge_to_id[old_layer]
        self.edge_list[edge_id] = new_layer
        self.edge_to_id[new_layer] = edge_id
        self.edge_to_id.pop(old_layer)
        self.old_edge_ids.pop(edge_id)

    def to_dense_deeper_model(self, target, param):
        pass

    def to_dense_wider_model(self, pre_layer, n_add):
        output_id = self.node_to_id[pre_layer.output]
        next_layer_list = self._search_following_conv(output_id)
        new_pre_layer, new_next_layer_list = dense_to_wider_layer(pre_layer,
                                                                  next_layer_list,
                                                                  n_add)
        for old_layer, new_layer in zip(next_layer_list, new_next_layer_list):
            self._replace_edge(old_layer, new_layer)
        self._replace_edge(pre_layer, new_pre_layer)
        return self.produce_model()

    def _topological_order(self):
        q = Queue()
        in_degree = {}
        for i in range(self.n_nodes):
            in_degree[i] = 0
        for u in range(self.n_nodes):
            for v, _ in self.adj_list[u]:
                in_degree[v] += 1
        for i in range(self.n_nodes):
            if in_degree[i] == 0:
                q.put(i)

        order_list = []
        while not q.empty():
            u = q.get()
            order_list.append(u)
            for v, _ in self.adj_list[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.put(v)
        return order_list
