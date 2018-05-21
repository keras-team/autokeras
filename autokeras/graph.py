from copy import deepcopy

from queue import Queue

from keras import Input
from keras.engine import Model
from keras.layers import Concatenate, Dense, BatchNormalization, Dropout, Activation, Flatten
from keras.regularizers import l2

from autokeras import constant
from autokeras.layer_transformer import wider_bn, wider_next_conv, wider_next_dense, wider_weighted_add, \
    wider_pre_dense, wider_pre_conv, deeper_conv_block, dense_to_deeper_block
from autokeras.layers import WeightedAdd, StubConcatenate, StubWeightedAdd
from autokeras.stub import to_stub_model
from autokeras.utils import get_int_tuple, is_layer, layer_width


class NetworkDescriptor:
    CONCAT_CONNECT = 'concat'
    ADD_CONNECT = 'add'

    def __init__(self):
        self.skip_connections = []
        self.conv_widths = []
        self.dense_widths = []

    @property
    def n_dense(self):
        return len(self.dense_widths)

    @property
    def n_conv(self):
        return len(self.conv_widths)

    def add_conv_width(self, width):
        self.conv_widths.append(width)

    def add_dense_width(self, width):
        self.dense_widths.append(width)

    def add_skip_connection(self, u, v, connection_type):
        if connection_type not in [self.CONCAT_CONNECT, self.ADD_CONNECT]:
            raise ValueError('connection_type should be NetworkDescriptor.CONCAT_CONNECT '
                             'or NetworkDescriptor.ADD_CONNECT.')
        self.skip_connections.append((u, v, connection_type))


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


class Graph:
    """A class represent the neural architecture graph of a Keras model.

    Graph extracts the neural architecture graph from a Keras model. Each node in the graph
    is a intermediate tensor between layers. Each layer is an edge in the graph.

    Notably, multiple edges may refer to the same layer. (e.g. WeightedAdd layer is adding
    two tensor into one tensor. So it is related to two edges.)

    Attributes:
        model: The Keras model, from which to extract the graph.
        node_list: A list of tensors, the indices of the list are the identifiers.
        layer_list: A list of Keras layers, the indices of the list are the identifiers.
        node_to_id: A dict instance mapping from tensors to their identifiers.
        layer_to_id: A dict instance mapping from Keras layers to their identifiers.
        layer_id_to_input_node_ids: A dict instance mapping from layer identifiers
            to their input nodes identifiers.
        adj_list: A two dimensional list. The adjacency list of the graph. The first dimension is
            identified by tensor identifiers. In each edge list, the elements are two-element tuples
            of (tensor identifier, layer identifier).
        reverse_adj_list: A reverse adjacent list in the same format as adj_list.
        next_vis: A boolean list marking whether a node has been visited or not.
        pre_vis: A boolean list marking whether a node has been visited or not.
        middle_layer_vis: A boolean list marking whether a node has been visited or not.
    """

    def __init__(self, model, weighted=True):
        model = to_stub_model(model, weighted)
        layers = model.layers[1:]
        self.weighted = weighted
        self.input = model.inputs[0]
        self.output = model.outputs[0]
        self.node_list = []
        self.layer_list = []
        # node id start with 0
        self.node_to_id = {}
        self.layer_to_id = {}
        self.layer_id_to_input_node_ids = {}
        self.layer_id_to_output_node_ids = {}
        self.adj_list = {}
        self.reverse_adj_list = {}
        self.operation_history = []

        self.next_vis = None
        self.pre_vis = None
        self.middle_layer_vis = None
        self.input_shape = model.input_shape

        # Add all nodes
        for layer in layers:
            if isinstance(layer.input, list):
                for temp_input in layer.input:
                    if temp_input not in self.node_list:
                        self._add_node(temp_input)
            else:
                if layer.input not in self.node_list:
                    self._add_node(layer.input)
            self._add_node(layer.output)

        # Add all edges
        for layer in layers:
            if isinstance(layer.input, list):
                for temp_input in layer.input:
                    self._add_edge(layer,
                                   self.node_to_id[temp_input],
                                   self.node_to_id[layer.output])
            else:
                self._add_edge(layer,
                               self.node_to_id[layer.input],
                               self.node_to_id[layer.output])

    def clear_operation_history(self):
        self.operation_history = []

    @property
    def n_nodes(self):
        """Return the number of nodes in the model."""
        return len(self.node_list)

    @property
    def n_layers(self):
        """Return the number of layers in the model."""
        return len(self.layer_list)

    def _add_node(self, node):
        """Add node to node list if it not in node list."""
        node_id = len(self.node_list)
        self.node_to_id[node] = node_id
        self.node_list.append(node)
        self.adj_list[node_id] = []
        self.reverse_adj_list[node_id] = []

    def _add_new_node(self):
        node_value = len(self.node_list)
        self._add_node(node_value)
        return self.node_to_id[node_value]

    def _add_edge(self, layer, input_id, output_id):
        """Add edge to the graph."""

        if layer in self.layer_to_id:
            layer_id = self.layer_to_id[layer]
            if input_id not in self.layer_id_to_input_node_ids[layer_id]:
                self.layer_id_to_input_node_ids[layer_id].append(input_id)
            if output_id not in self.layer_id_to_output_node_ids[layer_id]:
                self.layer_id_to_output_node_ids[layer_id].append(output_id)
        else:
            layer_id = len(self.layer_list)
            self.layer_list.append(layer)
            self.layer_to_id[layer] = layer_id
            self.layer_id_to_input_node_ids[layer_id] = [input_id]
            self.layer_id_to_output_node_ids[layer_id] = [output_id]

        self.adj_list[input_id].append((output_id, layer_id))
        self.reverse_adj_list[output_id].append((input_id, layer_id))

    def _redirect_edge(self, u_id, v_id, new_v_id):
        """Redirect the edge to a new node.
        Change the edge originally from u_id to v_id into an edge from u_id to new_v_id
        while keeping all other property of the edge the same.
        """
        layer_id = None
        for index, edge_tuple in enumerate(self.adj_list[u_id]):
            if edge_tuple[0] == v_id:
                layer_id = edge_tuple[1]
                self.adj_list[u_id][index] = (new_v_id, layer_id)
                break

        for index, edge_tuple in enumerate(self.reverse_adj_list[v_id]):
            if edge_tuple[0] == u_id:
                layer_id = edge_tuple[1]
                self.reverse_adj_list[v_id].remove(edge_tuple)
                break
        self.reverse_adj_list[new_v_id].append((u_id, layer_id))
        for index, value in enumerate(self.layer_id_to_output_node_ids[layer_id]):
            if value == v_id:
                self.layer_id_to_output_node_ids[layer_id][index] = new_v_id
                break

    def _replace_layer(self, layer_id, new_layer):
        """Replace the layer with a new layer."""
        old_layer = self.layer_list[layer_id]
        self.layer_list[layer_id] = new_layer
        self.layer_to_id[new_layer] = layer_id
        self.layer_to_id.pop(old_layer)

    def _topological_order(self):
        """Return the topological order of the node ids."""
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

    def _get_pooling_layers(self, start_node_id, end_node_id):
        layer_list = []
        node_list = [start_node_id]
        self._depth_first_search(end_node_id, layer_list, node_list)
        return filter(lambda layer_id: is_layer(self.layer_list[layer_id], 'Pooling'), layer_list)

    def _depth_first_search(self, target_id, layer_id_list, node_list):
        u = node_list[-1]
        if u == target_id:
            return True

        for v, layer_id in self.adj_list[u]:
            layer_id_list.append(layer_id)
            node_list.append(v)
            if self._depth_first_search(target_id, layer_id_list, node_list):
                return True
            layer_id_list.pop()
            node_list.pop()

        return False

    def _search_next(self, u, start_dim, total_dim, n_add):
        """Search downward the graph for widening the layers.

        Args:
            u: The starting node identifier.
            start_dim: The dimension to insert the additional dimensions.
            total_dim: The total number of dimensions the layer has before widening.
            n_add: The number of dimensions to add.
        """
        for v, layer_id in self.adj_list[u]:
            layer = self.layer_list[layer_id]

            if is_layer(layer, 'Conv'):
                new_layer = wider_next_conv(layer, start_dim, total_dim, n_add, self.weighted)
                self._replace_layer(layer_id, new_layer)

            elif is_layer(layer, 'Dense'):
                new_layer = wider_next_dense(layer, start_dim, total_dim, n_add, self.weighted)
                self._replace_layer(layer_id, new_layer)

            elif is_layer(layer, 'BatchNormalization'):
                new_layer = wider_bn(layer, start_dim, total_dim, n_add, self.weighted)
                self._replace_layer(layer_id, new_layer)
                self._search_next(v, start_dim, total_dim, n_add)

            elif is_layer(layer, 'WeightedAdd'):
                new_layer = wider_weighted_add(layer, n_add, self.weighted)
                self._replace_layer(layer_id, new_layer)
                self._search_next(v, start_dim, total_dim, n_add)

            elif is_layer(layer, 'Concatenate'):
                if self.layer_id_to_input_node_ids[layer_id][1] == u:
                    # u is on the right of the concat
                    # next_start_dim += next_total_dim - total_dim
                    left_dim = self._upper_layer_width(self.layer_id_to_input_node_ids[layer_id][0])
                    next_start_dim = start_dim + left_dim
                    next_total_dim = total_dim + left_dim
                else:
                    next_start_dim = start_dim
                    next_total_dim = total_dim + self._upper_layer_width(self.layer_id_to_input_node_ids[layer_id][1])
                self._search_next(v, next_start_dim, next_total_dim, n_add)

            else:
                self._search_next(v, start_dim, total_dim, n_add)

    def _search_pre(self, u, start_dim, total_dim, n_add):
        """Search upward the graph for widening the layers.

        Args:
            u: The starting node identifier.
            start_dim: The dimension to insert the additional dimensions.
            total_dim: The total number of dimensions the layer has before widening.
            n_add: The number of dimensions to add.
        """
        if self.pre_vis[u]:
            return
        self.pre_vis[u] = True
        self._search_next(u, start_dim, total_dim, n_add)
        for v, layer_id in self.reverse_adj_list[u]:
            layer = self.layer_list[layer_id]
            if is_layer(layer, 'Conv'):
                new_layer = wider_pre_conv(layer, n_add, self.weighted)
                self._replace_layer(layer_id, new_layer)
            elif is_layer(layer, 'Dense'):
                new_layer = wider_pre_dense(layer, n_add, self.weighted)
                self._replace_layer(layer_id, new_layer)
            elif is_layer(layer, 'BatchNormalization'):
                self._search_pre(v, start_dim, total_dim, n_add)
            elif is_layer(layer, 'Concatenate'):
                if self.layer_id_to_input_node_ids[layer_id][1] == v:
                    # v is on the right
                    other_branch_v = self.layer_id_to_input_node_ids[layer_id][0]
                    if self.pre_vis[other_branch_v]:
                        # The other branch is already been widen, which means the widen for upper part of this concat
                        #  layer is done.
                        continue
                    pre_total_dim = self._upper_layer_width(v)
                    pre_start_dim = start_dim - (total_dim - pre_total_dim)
                    self._search_pre(v, pre_start_dim, pre_total_dim, n_add)
            else:
                self._search_pre(v, start_dim, total_dim, n_add)

    def _upper_layer_width(self, u):
        for v, layer_id in self.reverse_adj_list[u]:
            layer = self.layer_list[layer_id]
            if is_layer(layer, 'Conv') or is_layer(layer, 'Dense'):
                return layer_width(layer)
            elif is_layer(layer, 'Concatenate'):
                a = self.layer_id_to_input_node_ids[layer_id][0]
                b = self.layer_id_to_input_node_ids[layer_id][1]
                return self._upper_layer_width(a) + self._upper_layer_width(b)
            else:
                return self._upper_layer_width(v)
        return self.input_shape[-1]

    def to_conv_deeper_model(self, target_id, kernel_size):
        """Insert a convolution, batch-normalization, relu block after the target block.

        Args:
            target_id: A convolutional layer ID. The new block should be inserted after the relu layer
                in its conv-batch-relu block.
            kernel_size: An integer. The kernel size of the new convolutional layer.

        Returns:
            A new Keras model with the inserted block.
        """
        self.operation_history.append(('to_conv_deeper_model', target_id, kernel_size))
        target = self.layer_list[target_id]
        new_layers = deeper_conv_block(target, kernel_size, self.weighted)
        output_id = self._conv_block_end_node(target_id)

        self._insert_new_layers(new_layers, output_id)

    def to_wider_model(self, pre_layer_id, n_add):
        """Widen the last dimension of the output of the pre_layer.

        Args:
            pre_layer_id: A convolutional layer or dense layer.
            n_add: The number of dimensions to add.

        Returns:
            A new Keras model with the widened layers.
        """
        self.operation_history.append(('to_wider_model', pre_layer_id, n_add))
        pre_layer = self.layer_list[pre_layer_id]
        output_id = self.layer_id_to_output_node_ids[pre_layer_id][0]
        dim = layer_width(pre_layer)
        if is_layer(pre_layer, 'Conv'):
            new_layer = wider_pre_conv(pre_layer, n_add, self.weighted)
            self._replace_layer(pre_layer_id, new_layer)
        else:
            new_layer = wider_pre_dense(pre_layer, n_add, self.weighted)
            self._replace_layer(pre_layer_id, new_layer)
        self._search_next(output_id, dim, dim, n_add)

    def to_dense_deeper_model(self, target_id):
        """Insert a dense layer after the target layer.

        Args:
            target_id: A dense layer.

        Returns:
            A new Keras model with an inserted dense layer.
        """
        self.operation_history.append(('to_dense_deeper_model', target_id))
        target = self.layer_list[target_id]
        new_layers = dense_to_deeper_block(target, self.weighted)
        output_id = self._dense_block_end_node(target_id)

        self._insert_new_layers(new_layers, output_id)

    def _insert_new_layers(self, new_layers, output_id):
        node_ids = []
        n_new_layers = len(new_layers)
        for i in range(n_new_layers):
            node_ids.append(self._add_new_node())
        for i in range(n_new_layers - 1):
            self._add_edge(new_layers[i], node_ids[i], node_ids[i + 1])
        self._add_edge(new_layers[n_new_layers - 1], node_ids[n_new_layers - 1], self.adj_list[output_id][0][0])
        self._redirect_edge(output_id, self.adj_list[output_id][0][0], node_ids[0])

    def _block_end_node(self, layer_id, block_size):
        ret = self.layer_id_to_output_node_ids[layer_id][0]
        for i in range(block_size - 2):
            ret = self.adj_list[ret][0][0]
        return ret

    def _dense_block_end_node(self, layer_id):
        return self._block_end_node(layer_id, constant.DENSE_BLOCK_DISTANCE)

    def _conv_block_end_node(self, layer_id):
        """

        Args:
            layer_id: the convolutional layer ID.

        Returns:
            The input node ID of the last layer in the convolutional block.

        """
        return self._block_end_node(layer_id, constant.CONV_BLOCK_DISTANCE)

    def to_add_skip_model(self, start_id, end_id):
        """Add a weighted add skip connection from before start node to end node.

        Args:
            start_id: The convolutional layer ID, after which to start the skip-connection.
            end_id: The convolutional layer ID, after which to end the skip-connection.

        Returns:
            A new Keras model with the added connection.
        """
        self.operation_history.append(('to_add_skip_model', start_id, end_id))
        conv_layer_ids = self._conv_layer_ids_in_order()
        start_id = conv_layer_ids[conv_layer_ids.index(start_id) + 1]
        conv_block_input_id = self.layer_id_to_input_node_ids[start_id][0]
        conv_block_input_id = self.reverse_adj_list[conv_block_input_id][0][0]
        conv_block_input_id = self.reverse_adj_list[conv_block_input_id][0][0]

        dropout_input_id = self._conv_block_end_node(end_id)

        # Add the pooling layer chain.
        pooling_layer_list = self._get_pooling_layers(conv_block_input_id, dropout_input_id)
        skip_output_id = conv_block_input_id
        for index, layer_id in enumerate(pooling_layer_list):
            layer = self.layer_list[layer_id]
            new_node_id = self._add_new_node()
            self._add_edge(deepcopy(layer), skip_output_id, new_node_id)
            skip_output_id = new_node_id

        # Add the weighted add layer.
        new_node_id = self._add_new_node()
        layer = StubWeightedAdd()
        if self.weighted:
            layer.set_weights(WeightedAdd().get_weights())

        dropout_output_id = self.adj_list[dropout_input_id][0][0]
        self._redirect_edge(dropout_input_id, dropout_output_id, new_node_id)
        self._add_edge(layer, new_node_id, dropout_output_id)
        self._add_edge(layer, skip_output_id, dropout_output_id)

    def to_concat_skip_model(self, start_id, end_id):
        """Add a weighted add concatenate connection from before start node to end node.

        Returns:
            A new Keras model with the added connection.
        """
        self.operation_history.append(('to_concat_skip_model', start_id, end_id))
        # start = self.layer_list[start_id]
        conv_layer_ids = self._conv_layer_ids_in_order()
        start_id = conv_layer_ids[conv_layer_ids.index(start_id) + 1]
        conv_block_input_id = self.layer_id_to_input_node_ids[start_id][0]
        conv_block_input_id = self.reverse_adj_list[conv_block_input_id][0][0]
        conv_block_input_id = self.reverse_adj_list[conv_block_input_id][0][0]

        end = self.layer_list[end_id]
        dropout_input_id = self._conv_block_end_node(end_id)

        # Add the pooling layer chain.
        pooling_layer_list = self._get_pooling_layers(conv_block_input_id, dropout_input_id)
        skip_output_id = conv_block_input_id
        for index, layer_id in enumerate(pooling_layer_list):
            layer = self.layer_list[layer_id]
            new_node_id = self._add_new_node()
            self._add_edge(deepcopy(layer), skip_output_id, new_node_id)
            skip_output_id = new_node_id

        # Add the concatenate layer.
        new_node_id = self._add_new_node()
        layer = StubConcatenate()

        dropout_output_id = self.adj_list[dropout_input_id][0][0]
        self._redirect_edge(dropout_input_id, dropout_output_id, new_node_id)
        self._add_edge(layer, new_node_id, dropout_output_id)
        self._add_edge(layer, skip_output_id, dropout_output_id)

        # Widen the related layers.
        dim = layer_width(end)
        n_add = self._upper_layer_width(conv_block_input_id)
        self._search_next(dropout_output_id, dim, dim, n_add)

    def extract_descriptor(self):
        ret = NetworkDescriptor()
        topological_node_list = self._topological_order()
        for u in topological_node_list:
            for v, layer_id in self.adj_list[u]:
                layer = self.layer_list[layer_id]
                if is_layer(layer, 'Conv'):
                    ret.add_conv_width(layer_width(layer))
                if is_layer(layer, 'Dense'):
                    ret.add_dense_width(layer_width(layer))

        # The position of each node, how many Conv and Dense layers before it.
        pos = [0] * len(topological_node_list)
        for v in topological_node_list:
            layer_count = 0
            for u, layer_id in self.reverse_adj_list[v]:
                layer = self.layer_list[layer_id]
                weighted = 0
                if is_layer(layer, 'Conv') or is_layer(layer, 'Dense'):
                    weighted = 1
                layer_count = max(pos[u] + weighted, layer_count)
            pos[v] = layer_count

        for u in topological_node_list:
            for v, layer_id in self.adj_list[u]:
                if pos[u] == pos[v]:
                    continue
                layer = self.layer_list[layer_id]
                if is_layer(layer, 'Concatenate'):
                    ret.add_skip_connection(pos[u], pos[v], NetworkDescriptor.CONCAT_CONNECT)
                if is_layer(layer, 'WeightedAdd'):
                    ret.add_skip_connection(pos[u], pos[v], NetworkDescriptor.ADD_CONNECT)

        return ret

    def produce_model(self):
        """Build a new Keras model based on the current graph."""
        input_tensor = Input(shape=get_int_tuple(self.input.shape[1:]))
        input_id = self.node_to_id[self.input]
        output_id = self.node_to_id[self.output]

        new_to_old_layer = {}

        node_list = deepcopy(self.node_list)
        node_list[input_id] = input_tensor

        node_to_id = deepcopy(self.node_to_id)
        node_to_id[input_tensor] = input_id

        for v in self._topological_order():
            for u, layer_id in self.reverse_adj_list[v]:
                layer = self.layer_list[layer_id]

                if isinstance(layer, (StubWeightedAdd, StubConcatenate)):
                    edge_input_tensor = list(map(lambda x: node_list[x],
                                                 self.layer_id_to_input_node_ids[layer_id]))
                else:
                    edge_input_tensor = node_list[u]

                new_layer = to_real_layer(layer)
                new_to_old_layer[new_layer] = layer

                temp_tensor = new_layer(edge_input_tensor)
                node_list[v] = temp_tensor
                node_to_id[temp_tensor] = v
        model = Model(input_tensor, node_list[output_id])
        for layer in model.layers[1:]:
            if not isinstance(layer, (Activation, Dropout, Concatenate)):
                old_layer = new_to_old_layer[layer]
                if self.weighted:
                    layer.set_weights(old_layer.get_weights())
        return model

    def _layer_ids_in_order(self, layer_ids):
        node_id_to_order_index = {}
        for index, node_id in enumerate(self._topological_order()):
            node_id_to_order_index[node_id] = index
        return sorted(layer_ids,
                      key=lambda layer_id:
                      node_id_to_order_index[self.layer_id_to_output_node_ids[layer_id][0]])

    def _layer_ids_by_type(self, type_str):
        return list(filter(lambda layer_id: is_layer(self.layer_list[layer_id], type_str), range(self.n_layers)))

    def _conv_layer_ids_in_order(self):
        return self._layer_ids_in_order(self._layer_ids_by_type('Conv'))

    def _dense_layer_ids_in_order(self):
        return self._layer_ids_in_order(self._layer_ids_by_type('Dense'))

    def deep_layer_ids(self):
        return self._conv_layer_ids_in_order() + self._dense_layer_ids_in_order()[:-1]

    def wide_layer_ids(self):
        return self._conv_layer_ids_in_order()[:-1] + self._dense_layer_ids_in_order()[:-1]

    def skip_connection_layer_ids(self):
        return self._conv_layer_ids_in_order()[:-1]
