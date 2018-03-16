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
    # The last conv layer cannot be widen since wider operator cannot be done over the two sides of flatten.
    conv_layers = list(filter(lambda x: is_conv_layer(x), graph.layer_list))[:-1]
    # The first layer cannot be widen since widen operator cannot be done over the two sides of flatten.
    # The last layer is softmax, which also cannot be widen.
    dense_layers = list(filter(lambda x: is_dense_layer(x), graph.layer_list))[1:-1]

    if len(dense_layers) == 0:
        weighted_layers = conv_layers
    elif randint(0, 1) == 0:
        weighted_layers = conv_layers
    else:
        weighted_layers = dense_layers

    print(weighted_layers)
    if len(weighted_layers) <= 1:
        target = weighted_layers[0]
    else:
        target = weighted_layers[randint(0, len(weighted_layers) - 1)]

    if is_conv_layer(target):
        n_add = randint(1, 4 * target.filters)
    else:
        n_add = randint(1, 4 * target.units)

    graph.to_wider_model(graph.layer_to_id[target], n_add)
    return graph


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
    # The last conv layer cannot be widen since wider operator cannot be done over the two sides of flatten.
    weighted_layers = list(filter(lambda x: is_conv_layer(x), graph.layer_list))[:-1]
    index_a = randint(0, len(weighted_layers) - 1)
    index_b = randint(0, len(weighted_layers) - 1)
    if index_a == index_b:
        if index_b == 0:
            index_a = index_b + 1
        else:
            index_a = index_b - 1
    if index_a > index_b:
        index_a, index_b = index_b, index_a
    a = weighted_layers[index_a]
    b = weighted_layers[index_b]
    a_id = graph.layer_to_id[a]
    b_id = graph.layer_to_id[b]
    if a.output_shape[-1] != b.output_shape[-1]:
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
    weighted_layers = list(filter(lambda x: isinstance(x, tuple(WEIGHTED_LAYER_FUNC_LIST)), graph.layer_list))[:-1]
    target = weighted_layers[randint(0, len(weighted_layers) - 1)]
    if is_conv_layer(target):
        graph.to_conv_deeper_model(graph.layer_to_id[target], randint(1, 2) * 2 + 1)
    else:
        graph.to_dense_deeper_model(graph.layer_to_id[target])
    return graph


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
            graphs.append(to_wider_graph(deepcopy(graph)))
        elif operation == 1:
            # deeper
            graphs.append(to_deeper_graph(deepcopy(graph)))
        elif operation == 2:
            # skip
            graphs.append(to_skip_connection_graph(deepcopy(graph)))

    return graphs
