from copy import deepcopy
from random import randint, random

from autokeras import constant
from autokeras.constant import WEIGHTED_LAYER_FUNC_LIST
from autokeras.utils import is_conv_layer, is_dense_layer


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
        n_add = randint(1, 4 * graph.layer_list[target_id].filters)
    else:
        n_add = randint(1, 4 * graph.layer_list[target_id].units)

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
    if graph.layer_list[a_id].filters != graph.layer_list[b_id].filters:
        graph.to_concat_skip_model(a_id, b_id)
    elif random() < 0.5:
        graph.to_add_skip_model(a_id, b_id)
    else:
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


def transform(graph):
    """Return new model after operations

    Args:
        graph: the model from which we get new model

    Returns:
        A list of graphs.
    """
    graphs = []
    for target_id in graph.wide_layer_ids():
        temp_graph = deepcopy(graph)
        if is_conv_layer(temp_graph.layer_list[target_id]):
            n_add = temp_graph.layer_list[target_id].filters
            temp_graph.to_wider_model(target_id, n_add)
        else:
            n_add = temp_graph.layer_list[target_id].units
            temp_graph.to_wider_model(target_id, n_add)
        graphs.append(temp_graph)

    for target_id in graph.deep_layer_ids():
        temp_graph = deepcopy(graph)
        if is_conv_layer(temp_graph.layer_list[target_id]):
            temp_graph.to_conv_deeper_model(target_id, randint(1, 2) * 2 + 1)
        else:
            temp_graph.to_dense_deeper_model(target_id)
        graphs.append(temp_graph)

    skip_ids = graph.skip_connection_layer_ids()
    for index_a, a_id in enumerate(skip_ids):
        temp_graph = deepcopy(graph)
        for b_id in skip_ids[index_a + 1:]:
            if temp_graph.layer_list[a_id].filters != temp_graph.layer_list[b_id].filters:
                temp_graph.to_concat_skip_model(a_id, b_id)
            else:
                temp_graph.to_add_skip_model(a_id, b_id)
            graphs.append(temp_graph)

    return graphs
