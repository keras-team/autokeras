from copy import deepcopy
from random import randint, randrange

from autokeras import constant
from autokeras.layers import is_conv_layer


def to_wider_graph(graph):
    weighted_layer_ids = graph.wide_layer_ids()
    if len(weighted_layer_ids) <= 1:
        target_id = weighted_layer_ids[0]
    else:
        target_id = weighted_layer_ids[randint(0, len(weighted_layer_ids) - 1)]

    if is_conv_layer(graph.layer_list[target_id]):
        n_add = graph.layer_list[target_id].filters
    else:
        n_add = graph.layer_list[target_id].units

    graph.to_wider_model(target_id, n_add)
    return graph


def to_skip_connection_graph(graph):
    # The last conv layer cannot be widen since wider operator cannot be done over the two sides of flatten.
    weighted_layer_ids = graph.skip_connection_layer_ids()
    index_a = randint(0, len(weighted_layer_ids) - 1)
    index_b = randint(0, len(weighted_layer_ids) - 1)
    if index_a == index_b:
        if index_b == 0:
            index_a = index_b + 1
        else:
            index_a = index_b - 1
    if index_a > index_b:
        index_a, index_b = index_b, index_a
    a_id = weighted_layer_ids[index_a]
    b_id = weighted_layer_ids[index_b]
    if graph.layer_list[a_id].filters == graph.layer_list[b_id].filters:
        graph.to_add_skip_model(a_id, b_id)
    else:
        graph.to_concat_skip_model(a_id, b_id)
    return graph


def to_deeper_graph(graph):
    weighted_layer_ids = graph.deep_layer_ids()
    target_id = weighted_layer_ids[randint(0, len(weighted_layer_ids) - 1)]
    if is_conv_layer(graph.layer_list[target_id]):
        graph.to_conv_deeper_model(target_id, randint(1, 2) * 2 + 1)
    else:
        graph.to_dense_deeper_model(target_id)
    return graph


def legal_graph(graph):
    descriptor = graph.extract_descriptor()
    skips = descriptor.skip_connections
    if len(skips) != len(set(skips)):
        return False
    return True


def transform(graph):
    graphs = []
    for i in range(constant.N_NEIGHBOURS):
        a = randrange(3)
        if a == 0:
            graphs.append(to_deeper_graph(deepcopy(graph)))
        elif a == 1:
            graphs.append(to_wider_graph(deepcopy(graph)))
        elif a == 2:
            graphs.append(to_skip_connection_graph(deepcopy(graph)))
    return list(filter(lambda graph: legal_graph(graph), graphs))


def default_transform(graph):
    graph = deepcopy(graph)
    graph.to_conv_deeper_model(1, 3)
    graph.to_conv_deeper_model(1, 3)
    graph.to_conv_deeper_model(6, 3)
    graph.to_conv_deeper_model(11, 3)
    graph.to_add_skip_model(1, 18)
    graph.to_add_skip_model(18, 26)
    graph.to_add_skip_model(26, 30)
    return [graph]
