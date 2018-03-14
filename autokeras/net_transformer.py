from copy import deepcopy
from random import randint, random

from autokeras import constant
from autokeras.constant import WEIGHTED_LAYER_FUNC_LIST
from autokeras.utils import is_conv_layer


def to_wider_graph(graph):
    """Return wider model

    Args:
        graph: the model from which we get wider model

    Returns:
        The wider model
    """
    weighted_layers = list(filter(lambda x: isinstance(x, tuple(WEIGHTED_LAYER_FUNC_LIST)), graph.layer_list))[:-1]
    target = weighted_layers[randint(0, len(weighted_layers) - 1)]
    if is_conv_layer(target):
        n_add = randint(1, 4 * target.filters)
    else:
        n_add = randint(1, 4 * target.units)
    graph.to_wider_model(graph.layer_to_id[target], n_add)
    return deepcopy(graph)


def copy_conv_graph(graph):
    """Return copied convolution model

    Args:
        graph: the model we want to copy

    Returns:
        The copied model
    """
    return deepcopy(graph)


def to_skip_connection_graph(graph):
    """Return skip_connected model

    Args:
        graph: the model from which we get skip_connected model

    Returns:
        The skip_connected model
    """
    weighted_layers = list(filter(lambda x: is_conv_layer(x), graph.layer_list))
    index_a = randint(0, len(weighted_layers) - 1)
    index_b = randint(0, len(weighted_layers) - 1)
    if index_a > index_b:
        index_a, index_b = index_b, index_a
    a = graph.layer_to_id[weighted_layers[index_a]]
    b = graph.layer_to_id[weighted_layers[index_b]]
    # TODO: Graph doesn't contain the shape information, but we need it for whether can add skip.
    if a.input.shape != b.output.shape:
        graph.to_concat_skip_model(a, b)
    elif random() < 0.5:
        graph.to_add_skip_model(a, b)
    else:
        graph.to_concat_skip_model(a, b)
    return deepcopy(graph)


def to_deeper_graph(graph):
    """Return deeper model

    Args:
        graph: the model from which we get deeper model

    Returns:
        The deeper model
    """
    weighted_layers = list(filter(lambda x: isinstance(x, tuple(WEIGHTED_LAYER_FUNC_LIST)), graph.layer_list))[:-1]
    target = weighted_layers[randint(0, len(weighted_layers) - 1)]
    if is_conv_layer(target):
        graph.to_conv_deeper_model(graph.layer_to_id[target], randint(1, 2) * 2 + 1)
    else:
        graph.to_dense_deeper_model(graph.layer_to_id[target])
    return deepcopy(graph)


def transform(graph):
    """Return new model after operations

    Args:
        graph: the model from which we get new model

    Returns:
        The new model
    """
    graphs = []
    for i in range(constant.N_NEIGHBORS):
        operation = randint(0, 2)

        if operation == 0:
            # wider
            graphs.append(to_wider_graph(graph))
        elif operation == 1:
            # deeper
            graphs.append(to_deeper_graph(graph))
        elif operation == 2:
            # skip
            graphs.append(to_skip_connection_graph(graph))

    return graphs
