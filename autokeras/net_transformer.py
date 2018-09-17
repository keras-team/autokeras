from copy import deepcopy
from operator import itemgetter
from random import randrange, sample

from autokeras.graph import NetworkDescriptor

from autokeras.constant import Constant
from autokeras.layers import is_layer


def to_wider_graph(graph):
    weighted_layer_ids = graph.wide_layer_ids()
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


def to_deeper_graph(graph):
    weighted_layer_ids = graph.deep_layer_ids()

    deeper_layer_ids = sample(weighted_layer_ids, 1)

    for layer_id in deeper_layer_ids:
        layer = graph.layer_list[layer_id]
        if is_layer(layer, 'Conv'):
            graph.to_conv_deeper_model(layer_id, 3)
        else:
            graph.to_dense_deeper_model(layer_id)
    return graph


def legal_graph(graph):
    descriptor = graph.extract_descriptor()
    skips = descriptor.skip_connections
    if len(skips) != len(set(skips)):
        return False
    return True


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

    return list(filter(lambda x: legal_graph(x), graphs))


def default_transform(graph):
    graph = deepcopy(graph)
    graph.to_conv_deeper_model(1, 3)
    graph.to_conv_deeper_model(1, 3)
    graph.to_conv_deeper_model(5, 3)
    graph.to_conv_deeper_model(9, 3)
    graph.to_add_skip_model(1, 18)
    graph.to_add_skip_model(18, 24)
    graph.to_add_skip_model(24, 27)
    return [graph]
