from random import randint

from keras.layers import Dense
from autokeras.graph import Graph
from autokeras.utils import is_conv_layer


def to_deeper_dense_model(model, target):
    graph = Graph(model)
    return graph.to_dense_deeper_model(target, randint(1, 2) * 2 + 1)


def to_wider_dense_model(model, target):
    graph = Graph(model)
    n_add = randint(1, 4 * target.units)
    return graph.to_dense_wider_model(target, n_add)


def to_deeper_conv_model(model, target):
    graph = Graph(model)
    return graph.to_conv_deeper_model(target, randint(1, 2) * 2 + 1)


def to_wider_conv_model(model, target):
    graph = Graph(model)
    n_add = randint(1, 4 * target.filters)
    return graph.to_conv_wider_model(target, n_add)


def copy_conv_model(model):
    graph = Graph(model)
    return graph.produce_model()


def to_skip_connection_model(conv_model):
    return conv_model


def transform(model):
    models = []

    for index, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            models.append(to_deeper_dense_model(model, layer))
            models.append(to_wider_dense_model(model, layer))
        elif is_conv_layer(layer):
            models.append(to_deeper_conv_model(model, layer))
            models.append(to_wider_conv_model(model, layer))

    models.append(to_skip_connection_model(model))

    return models
