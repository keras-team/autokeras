from queue import Queue

from keras import Input
from keras.engine import Model
from keras.layers import Concatenate

from autokeras.layer_transformer import *
from autokeras.layers import WeightedAdd
from autokeras.utils import copy_layer, is_conv_layer, get_int_tuple, is_pooling_layer


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
        old_layer_ids: A dict instance stores the identifiers of the not updated layers.
        adj_list: A two dimensional list. The adjacency list of the graph. The first dimension is
            identified by tensor identifiers. In each edge list, the elements are two-element tuples
            of (tensor identifier, layer identifier).
        reverse_adj_list: A reverse adjacent list in the same format as adj_list.
        next_vis: A boolean list marking whether a node has been visited or not.
        pre_vis: A boolean list marking whether a node has been visited or not.
        middle_layer_vis: A boolean list marking whether a node has been visited or not.
    """
    def __init__(self, model):
        layers = model.layers[1:]
        self.model = model
        self.node_list = []
        self.layer_list = []
        # node id start with 0
        self.node_to_id = {}
        self.layer_to_id = {}
        self.layer_id_to_input_node_ids = {}
        self.old_layer_ids = {}
        self.adj_list = {}
        self.reverse_adj_list = {}

        self.next_vis = None
        self.pre_vis = None
        self.middle_layer_vis = None

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
        """Return the number of nodes in the model."""
        return len(self.node_list)

    @property
    def n_layers(self):
        """Return the number of layers in the model."""
        return len(self.layer_list)

    def _add_node(self, node):
        """Add node to node list if it not in node list."""
        if node not in self.node_list:
            node_id = len(self.node_list)
            self.node_to_id[node] = node_id
            self.node_list.append(node)
            self.adj_list[node_id] = []
            self.reverse_adj_list[node_id] = []

    def _add_edge(self, layer, input_id=None, output_id=None, old=True):
        """Add edge to the graph."""
        if input_id is None:
            if isinstance(layer.input, list):
                for temp_input in layer.input:
                    input_id = self.node_to_id[temp_input]
                    self._add_edge(layer, input_id, output_id, old)
                return
            input_id = self.node_to_id[layer.input]

        if output_id is None:
            output_id = self.node_to_id[layer.output]

        if layer in self.layer_to_id:
            layer_id = self.layer_to_id[layer]
            self.layer_id_to_input_node_ids[layer_id].append(input_id)
        else:
            layer_id = len(self.layer_list)
            self.layer_list.append(layer)
            self.layer_to_id[layer] = layer_id
            self.layer_id_to_input_node_ids[layer_id] = [input_id]

        self.adj_list[input_id].append((output_id, layer_id))
        self.reverse_adj_list[output_id].append((input_id, layer_id))

        if old:
            self.old_layer_ids[layer_id] = True

    def _redirect_edge(self, u_id, v_id, new_v_id):
        """Redirect the edge to a new node.
        Change the edge originally from u_id to v_id into an edge from u_id to new_v_id
        while keeping all other property of the edge the same.
        """
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

        self._add_edge(self.layer_list[layer_id], u_id, new_v_id)

    def _search_next(self, u, start_dim, total_dim, n_add):
        """Search downward the graph for widening the layers.

        Args:
            u: The starting node identifier.
            start_dim: The dimension to insert the additional dimensions.
            total_dim: The total number of dimensions the layer has before widening.
            n_add: The number of dimensions to add.
        """
        if self.next_vis[u]:
            return
        self.next_vis[u] = True
        self._search_pre(u, start_dim, total_dim, n_add)
        for v, layer_id in self.adj_list[u]:
            layer = self.layer_list[layer_id]

            if is_conv_layer(layer):
                new_layer = wider_next_conv(layer, start_dim, total_dim, n_add)
                self._replace_layer(layer_id, new_layer)

            elif isinstance(layer, Dense):
                new_layer = wider_next_dense(layer, start_dim, total_dim, n_add)
                self._replace_layer(layer_id, new_layer)

            elif isinstance(layer, BatchNormalization):
                if not self.middle_layer_vis[layer_id]:
                    self.middle_layer_vis[layer_id] = True
                    new_layer = wider_bn(layer, start_dim, total_dim, n_add)
                    self._replace_layer(layer_id, new_layer)
                self._search_next(v, start_dim, total_dim, n_add)

            elif isinstance(layer, WeightedAdd):
                if not self.middle_layer_vis[layer_id]:
                    self.middle_layer_vis[layer_id] = True
                    new_layer = wider_weighted_add(layer, n_add)
                    self._replace_layer(layer_id, new_layer)
                self._search_next(v, start_dim, total_dim, n_add)

            elif isinstance(layer, Concatenate):
                next_start_dim = start_dim
                next_total_dim = layer.output_shape[-1]
                if self.node_list[u] is layer.input[1]:
                    # u is on the right of the concat
                    next_start_dim += next_total_dim - total_dim
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
            if is_conv_layer(layer):
                new_layer = wider_pre_conv(layer, n_add)
                self._replace_layer(layer_id, new_layer)
            elif isinstance(layer, Dense):
                new_layer = wider_pre_dense(layer, n_add)
                self._replace_layer(layer_id, new_layer)
            elif isinstance(layer, BatchNormalization):
                self._search_pre(v, start_dim, total_dim, n_add)
            elif isinstance(layer, Concatenate):
                if self.node_list[v] is layer.input[1]:
                    # v is on the right
                    pre_total_dim = layer.input_shape[1][-1]
                    pre_start_dim = start_dim - (total_dim - pre_total_dim)
                    self._search_pre(v, pre_start_dim, pre_total_dim, n_add)
            else:
                self._search_pre(v, start_dim, total_dim, n_add)

    def _replace_layer(self, layer_id, new_layer):
        """Replace the layer with a new layer."""
        old_layer = self.layer_list[layer_id]
        self.layer_list[layer_id] = new_layer
        self.layer_to_id[new_layer] = layer_id
        self.layer_to_id.pop(old_layer)
        self.old_layer_ids.pop(layer_id, None)

    def _topological_order(self):
        """Return the topological order of the nodes."""
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

    def to_add_skip_model(self, start, end):
        """Add a weighted add skip connection from start node to end node.

        Returns:
            A new Keras model with the added connection.
        """
        conv_input_id = self.node_to_id[start.input]
        relu_input_id = self.adj_list[self.node_to_id[end.output]][0][0]

        # Add the pooling layer chain.
        pooling_layer_list = self.get_pooling_layers(conv_input_id, relu_input_id)
        skip_output_id = conv_input_id
        for index, layer_id in enumerate(pooling_layer_list):
            layer = self.layer_list[layer_id]
            self._add_node(index)
            new_node_id = self.node_to_id[index]
            self._add_edge(copy_layer(layer), skip_output_id, new_node_id, False)
            skip_output_id = new_node_id

        # Add the weighted add layer.
        self._add_node('a')
        new_node_id = self.node_to_id['a']
        layer = WeightedAdd()
        single_input_shape = get_int_tuple(start.output_shape)
        layer.build([single_input_shape, single_input_shape])

        relu_output_id = self.adj_list[relu_input_id][0][0]
        self._redirect_edge(relu_input_id, relu_output_id, new_node_id)
        self._add_edge(layer, new_node_id, relu_output_id, False)
        self._add_edge(layer, skip_output_id, relu_output_id, False)

        return self.produce_model()

    def to_concat_skip_model(self, start, end):
        """Add a weighted add concatenate connection from start node to end node.

        Returns:
            A new Keras model with the added connection.
        """
        conv_input_id = self.node_to_id[start.input]
        relu_input_id = self.adj_list[self.node_to_id[end.output]][0][0]

        # Add the pooling layer chain.
        pooling_layer_list = self.get_pooling_layers(conv_input_id, relu_input_id)
        skip_output_id = conv_input_id
        for index, layer_id in enumerate(pooling_layer_list):
            layer = self.layer_list[layer_id]
            self._add_node(index)
            new_node_id = self.node_to_id[index]
            self._add_edge(copy_layer(layer), skip_output_id, new_node_id, False)
            skip_output_id = new_node_id

        # Add the weighted add layer.
        self._add_node('a')
        new_node_id = self.node_to_id['a']
        layer = Concatenate()
        left_input_shape = get_int_tuple(end.output_shape)
        right_input_shape = np.concatenate((left_input_shape[:-1], get_int_tuple(start.output_shape[-1:])))
        layer.build([left_input_shape, right_input_shape])

        relu_output_id = self.adj_list[relu_input_id][0][0]
        self._redirect_edge(relu_input_id, relu_output_id, new_node_id)
        self._add_edge(layer, new_node_id, relu_output_id, False)
        self._add_edge(layer, skip_output_id, relu_output_id, False)

        # Widen the related layers.
        self.next_vis = [False] * self.n_nodes
        self.pre_vis = [False] * self.n_nodes
        self.middle_layer_vis = [False] * len(self.layer_list)

        self.pre_vis[relu_output_id] = True
        dim = get_int_tuple(end.output_shape)[-1]
        n_add = get_int_tuple(start.output_shape)[-1]
        self._search_next(relu_output_id, dim, dim, n_add)
        return self.produce_model()

    def to_conv_deeper_model(self, target, kernel_size):
        """Insert a convolution, batch-normalization, relu block after the target block.

        Args:
            target: A convolutional layer. The new block should be inserted after the relu layer
                in its conv-batch-relu block.
            kernel_size: An integer. The kernel size of the new convolutional layer.

        Returns:
            A new Keras model with the inserted block.
        """
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

    def to_wider_model(self, pre_layer, n_add):
        """Widen the last dimension of the output of the pre_layer.

        Args:
            pre_layer: A convolutional layer or dense layer.
            n_add: The number of dimensions to add.

        Returns:
            A new Keras model with the widened layers.
        """
        output_id = self.node_to_id[pre_layer.output]
        self.next_vis = [False] * self.n_nodes
        self.pre_vis = [False] * self.n_nodes
        self.middle_layer_vis = [False] * len(self.layer_list)
        dim = get_int_tuple(pre_layer.output_shape)[-1]
        self._search_next(output_id, dim, dim, n_add)
        return self.produce_model()

    def to_dense_deeper_model(self, target):
        """Insert a dense layer after the target layer.

        Args:
            target: A dense layer.

        Returns:
            A new Keras model with an inserted dense layer.
        """
        new_layers = dense_to_deeper_layer(target)
        old_input_id = self.node_to_id[target.input]
        old_output_id = self.node_to_id[target.output]

        self._add_node(0)
        new_node_id = self.node_to_id[0]
        self._add_edge(new_layers, new_node_id, old_output_id, False)
        self._redirect_edge(old_input_id, old_output_id, new_node_id)
        return self.produce_model()

    def produce_model(self):
        """Build a new Keras model based on the current graph."""
        input_tensor = Input(shape=get_int_tuple(self.model.inputs[0].shape[1:]))
        input_id = self.node_to_id[self.model.inputs[0]]
        output_id = self.node_to_id[self.model.outputs[0]]

        id_to_tensor = {input_id: input_tensor}
        for v in self._topological_order():
            for u, layer_id in self.reverse_adj_list[v]:
                layer = self.layer_list[layer_id]

                if isinstance(layer, (WeightedAdd, Concatenate)):
                    edge_input_tensor = list(map(lambda x: id_to_tensor[x], self.layer_id_to_input_node_ids[layer_id]))
                else:
                    edge_input_tensor = id_to_tensor[u]

                if layer_id in self.old_layer_ids:
                    new_layer = copy_layer(layer)
                else:
                    new_layer = layer

                temp_tensor = new_layer(edge_input_tensor)
                id_to_tensor[v] = temp_tensor
        return Model(input_tensor, id_to_tensor[output_id])

    def get_pooling_layers(self, start_node_id, end_node_id):
        layer_list = []
        node_list = [start_node_id]
        self._depth_first_search(end_node_id, layer_list, node_list)
        return filter(lambda layer_id: is_pooling_layer(self.layer_list[layer_id]), layer_list)

    def _depth_first_search(self, target_id, layer_list, node_list):
        u = node_list[-1]
        if u == target_id:
            return True

        for v, layer_id in self.adj_list[u]:
            layer_list.append(layer_id)
            node_list.append(v)
            if self._depth_first_search(target_id, layer_list, node_list):
                return True
            layer_list.pop()
            node_list.pop()

        return False
