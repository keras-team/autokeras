from copy import deepcopy
from operator import itemgetter
from random import randrange, sample

from autokeras.nn.graph import NetworkDescriptor

from autokeras.constant import Constant
from autokeras.nn.layer_transformer import init_dense_weight, init_conv_weight, init_bn_weight
from autokeras.nn.layers import is_layer, StubDense, get_dropout_class, StubReLU, get_conv_class, \
    get_batch_norm_class, get_pooling_class


def to_wider_graph(graph):
    weighted_layer_ids = graph.wide_layer_ids()
    weighted_layer_ids = list(filter(lambda x: graph.layer_list[x].output.shape[-1], weighted_layer_ids))
    wider_layers = sample(weighted_layer_ids, 1)

    for layer_id in wider_layers:
        layer = graph.layer_list[layer_id]
        if is_layer(layer, 'Conv'):
            n_add = layer.filters
        else:
            n_add = layer.units

        graph.to_wider_model(layer_id, n_add)
    return graph


def to_skip_connection_graph(graph):
    # The last conv layer cannot be widen since wider operator cannot be done over the two sides of flatten.
    weighted_layer_ids = graph.skip_connection_layer_ids()
    descriptor = graph.extract_descriptor()
    sorted_skips = sorted(descriptor.skip_connections, key=itemgetter(2, 0, 1))
    p = 0
    valid_connection = []
    for skip_type in sorted([NetworkDescriptor.ADD_CONNECT, NetworkDescriptor.CONCAT_CONNECT]):
        for index_a in range(len(weighted_layer_ids)):
            for index_b in range(len(weighted_layer_ids))[index_a + 1:]:
                if p < len(sorted_skips) and sorted_skips[p] == (index_a + 1, index_b + 1, skip_type):
                    p += 1
                else:
                    valid_connection.append((index_a, index_b, skip_type))

    if len(valid_connection) < 1:
        return graph
    # n_skip_connection = randint(1, len(valid_connection))
    # for index_a, index_b, skip_type in sample(valid_connection, n_skip_connection):
    for index_a, index_b, skip_type in sample(valid_connection, 1):
        a_id = weighted_layer_ids[index_a]
        b_id = weighted_layer_ids[index_b]
        if skip_type == NetworkDescriptor.ADD_CONNECT:
            graph.to_add_skip_model(a_id, b_id)
        else:
            graph.to_concat_skip_model(a_id, b_id)
    return graph


def create_new_layer(input_shape, n_dim):
    dense_deeper_classes = [StubDense, get_dropout_class(n_dim), StubReLU]
    conv_deeper_classes = [get_conv_class(n_dim), get_batch_norm_class(n_dim), StubReLU, get_pooling_class(n_dim)]
    if len(input_shape) == 1:
        # It is in the dense layer part.
        layer_class = sample(dense_deeper_classes, 1)[0]
    else:
        # It is in the conv layer part.
        layer_class = sample(conv_deeper_classes, 1)[0]

    if layer_class == StubDense:
        new_layer = StubDense(input_shape[0], input_shape[0])

    elif layer_class == get_dropout_class(n_dim):
        new_layer = layer_class(Constant.DENSE_DROPOUT_RATE)

    elif layer_class == get_conv_class(n_dim):
        new_layer = layer_class(input_shape[-1], input_shape[-1], sample((1, 3, 5), 1)[0], stride=sample((1, 2), 1)[0])

    elif layer_class == get_batch_norm_class(n_dim):
        new_layer = layer_class(input_shape[-1])

    elif layer_class == get_pooling_class(n_dim):
        new_layer = layer_class(sample((1, 3, 5), 1)[0])

    else:
        new_layer = layer_class()

    return new_layer


def to_deeper_graph(graph):
    weighted_layer_ids = graph.deep_layer_ids()
    if len(weighted_layer_ids) >= Constant.MAX_LAYERS:
        return None

    deeper_layer_ids = sample(weighted_layer_ids, 1)

    for layer_id in deeper_layer_ids:
        layer = graph.layer_list[layer_id]
        new_layer = create_new_layer(layer.output.shape, graph.n_dim)
        graph.to_deeper_model(layer_id, new_layer)
    return graph


def transform(graph):
    graphs = []
    for i in range(Constant.N_NEIGHBOURS * 2):
        a = randrange(3)
        temp_graph = None
        if a == 0:
            temp_graph = to_deeper_graph(deepcopy(graph))
        elif a == 1:
            temp_graph = to_wider_graph(deepcopy(graph))
        elif a == 2:
            temp_graph = to_skip_connection_graph(deepcopy(graph))

        if temp_graph is not None and temp_graph.size() <= Constant.MAX_MODEL_SIZE:
            graphs.append(temp_graph)

        if len(graphs) >= Constant.N_NEIGHBOURS:
            break

    return graphs
