from random import randint, random

from autokeras import constant
from autokeras.constant import WEIGHTED_LAYER_FUNC_LIST
from autokeras.graph import Graph
from autokeras.utils import is_conv_layer


def to_wider_model(model):
    """Return wider model

    Args:
        model: the model from which we get wider model

    Returns:
        The wider model
    """
    graph = Graph(model)
    weighted_layers = list(filter(lambda x: isinstance(x, tuple(WEIGHTED_LAYER_FUNC_LIST)), model.layers))[:-1]
    target = weighted_layers[randint(0, len(weighted_layers) - 1)]
    if is_conv_layer(target):
        n_add = randint(1, 4 * target.filters)
    else:
        n_add = randint(1, 4 * target.units)
    return graph.to_wider_model(target, n_add)


def copy_conv_model(model):
    """Return copied convolution model

    Args:
        model: the model we want to copy

    Returns:
        The copied model
    """
    graph = Graph(model)
    return graph.produce_model()


def to_skip_connection_model(model):
    """Return skip_connected model

    Args:
        model: the model from which we get skip_connected model

    Returns:
        The skip_connected model
    """
    graph = Graph(model)
    weighted_layers = list(filter(lambda x: is_conv_layer(x), model.layers))
    index_a = randint(0, len(weighted_layers) - 1)
    index_b = randint(0, len(weighted_layers) - 1)
    if index_a > index_b:
        index_a, index_b = index_b, index_a
    a = weighted_layers[index_a]
    b = weighted_layers[index_b]
    if a.input.shape == b.output.shape:
        return graph.to_add_skip_model(a, b)
    elif random() < 0.5:
        return graph.to_add_skip_model(a, b)
    else:
        return graph.to_concat_skip_model(a, b)


def to_deeper_model(model):
    """Return deeper model

    Args:
        model: the model from which we get deeper model

    Returns:
        The deeper model
    """
    graph = Graph(model)
    weighted_layers = list(filter(lambda x: isinstance(x, tuple(WEIGHTED_LAYER_FUNC_LIST)), model.layers))[:-1]
    target = weighted_layers[randint(0, len(weighted_layers) - 1)]
    if is_conv_layer(target):
        return graph.to_conv_deeper_model(target, randint(1, 2) * 2 + 1)
    return graph.to_dense_deeper_model(target)


def transform(model):
    """Return new model after operations

    Args:
        model: the model from which we get new model

    Returns:
        The new model
    """
    models = []
    for i in range(constant.N_NEIGHBORS):
        operation = randint(0, 2)

        if operation == 0:
            # wider
            models.append(to_wider_model(model))
        elif operation == 1:
            # deeper
            models.append(to_deeper_model(model))
        elif operation == 2:
            # skip
            models.append(to_skip_connection_model(model))

    return models
