from copy import deepcopy
from random import randint, random, shuffle, randrange

from autokeras import constant
from autokeras.layers import is_conv_layer, is_dense_layer, WEIGHTED_LAYER_FUNC_LIST


def to_wider_graph(graph):
    """Return wider model

    Args:
        graph: the model from which we get wider model

    Returns:
        The wider model
    """
    weighted_layer_ids = graph.wide_layer_ids()
    if len(weighted_layer_ids) <= 1:
        target_id = weighted_layer_ids[0]
    else:
        target_id = weighted_layer_ids[randint(0, len(weighted_layer_ids) - 1)]

    if is_conv_layer(graph.layer_list[target_id]):
        n_add = randint(1, 2 * graph.layer_list[target_id].filters)
    else:
        n_add = randint(1, 2 * graph.layer_list[target_id].units)

    graph.to_wider_model(target_id, n_add)
    return graph


def to_skip_connection_graph(graph):
    """Return skip_connected model

    Args:
        graph: the model from which we get skip_connected model

    Returns:
        The skip_connected model
    """
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
    graph.to_concat_skip_model(a_id, b_id)
    return graph


def to_deeper_graph(graph):
    """Return deeper model

    Args:
        graph: the model from which we get deeper model

    Returns:
        The deeper model
    """
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
    """Return new model after operations

    Args:
        graph: the model from which we get new model

    Returns:
        A list of graphs.
    """
    graphs = []
    for i in range(8):
        a = randrange(3)
        if a == 0:
            graphs.append(to_deeper_graph(deepcopy(graph)))
        elif a == 1:
            graphs.append(to_wider_graph(deepcopy(graph)))
        elif a == 2:
            graphs.append(to_skip_connection_graph(deepcopy(graph)))
    return graphs

