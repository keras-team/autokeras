from random import randint, random

from autokeras import constant
from autokeras.constant import WEIGHTED_LAYER_FUNC_LIST
from autokeras.graph import Graph
from autokeras.utils import is_conv_layer


def to_wider_model(model):
    graph = Graph(model)
    weighted_layers = list(filter(lambda x: isinstance(x, tuple(WEIGHTED_LAYER_FUNC_LIST)), model.layers))[:-1]
    target = weighted_layers[randint(0, len(weighted_layers) - 1)]
    if is_conv_layer(target):
        n_add = randint(1, 4 * target.filters)
    else:
        n_add = randint(1, 4 * target.units)
    return graph.to_wider_model(target, n_add)


def copy_conv_model(model):
    graph = Graph(model)
    return graph.produce_model()


def to_skip_connection_model(model):
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
    graph = Graph(model)
    weighted_layers = list(filter(lambda x: isinstance(x, tuple(WEIGHTED_LAYER_FUNC_LIST)), model.layers))[:-1]
    target = weighted_layers[randint(0, len(weighted_layers) - 1)]
    if is_conv_layer(target):
        return graph.to_dense_deeper_model(target)
    return graph.to_conv_deeper_model(target, randint(1, 2) * 2 + 1)


def transform(model):
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
