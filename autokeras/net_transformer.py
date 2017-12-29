from random import randint

from keras import Input
from keras.engine import Model

from autokeras.graph import Graph
from autokeras.layer_transformer import dense_to_wider_layer, dense_to_deeper_layer, deeper_conv_block

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Conv3D

from autokeras.utils import copy_layer, is_conv_layer


def get_next_dense_conv(start, layers):
    """Start to search next dense or conv.

    long description.

    :param start:
    :param layers:
    :return:
    """
    new_next_wider_layer = None
    ind = None
    for j in range(start + 1, len(layers)):
        if isinstance(layers[j], (
                Dense, Conv1D, Conv2D, Conv3D)):
            new_next_wider_layer = layers[j]
            ind = j
            break
    if new_next_wider_layer is None:
        raise ValueError("There must be a Corresponding Dense or Convolution Layer")
    return new_next_wider_layer, ind


def replace_layers(model, level_list, layer_list):
    new_model = Sequential()
    new_layer_list = []

    for layer in model.layers:
        new_layer_list.append(copy_layer(layer))

    for index, level in enumerate(level_list):
        new_layer_list[level] = layer_list[index]

    for index, layer in enumerate(new_layer_list):
        new_model.add(layer)

    for index, layer in enumerate(new_model.layers):
        if index not in level_list:
            layer.set_weights(model.layers[index].get_weights())

    return new_model


def insert_layer(model, level, new_layer):
    new_model = Sequential()

    for index, layer in enumerate(model.layers):
        new_model.add(copy_layer(layer))
        new_model.layers[-1].set_weights(layer.get_weights())
        if index == level:
            new_model.add(new_layer)
    return new_model


def to_deeper_dense_model(model, level):
    new_deeper_layer_list = dense_to_deeper_layer(model.layers[level])
    return insert_layer(model, level, new_deeper_layer_list)


def to_wider_dense_model(model, level):
    next_wider_layer, ind = get_next_dense_conv(level, model.layers)
    n_add = randint(1, 4 * model.layers[level].units)
    new_wider_layer, new_next_wider_layer = dense_to_wider_layer(model.layers[level], next_wider_layer, n_add)
    return replace_layers(model, [level, ind], [new_wider_layer, new_next_wider_layer])


def to_deeper_conv_model(model, target):
    graph = Graph(model)
    return graph.to_deeper_model(target, randint(1, 2) * 2 + 1)


def to_wider_conv_model(model, target):
    graph = Graph(model)
    n_add = randint(1, 4 * target.filters)
    return graph.to_wider_model(target, n_add)


def copy_conv_model(model):
    node_old_to_new = {}
    new_model_input = Input(model.input_shape)
    node_old_to_new[model.inputs] = new_model_input

    for layer in model.layers:
        new_layer = copy_layer(layer)
        old_input = layer.input
        old_output = layer.output
        new_input = node_old_to_new[old_input]
        new_output = new_layer(new_input)
        node_old_to_new[old_output] = new_output
    return Model(new_model_input, node_old_to_new[model.outputs])


def to_skip_connection_model(conv_model):
    pass


def transform(model):
    models = []
    conv_model = model.layers[0]
    dense_model = model.layers[1]

    for index, layer in enumerate(dense_model):
        # search dense variation
        if isinstance(layer, Dense):
            models.append(Sequential(copy_conv_model(conv_model), to_deeper_dense_model(dense_model, index)))
            models.append(Sequential(copy_conv_model(conv_model), to_wider_dense_model(dense_model, index)))

    for layer in conv_model.layers:
        # search conv variation
        if is_conv_layer(layer):
            models.append(Sequential(to_deeper_conv_model(conv_model, layer), dense_model))
            models.append(Sequential(to_wider_conv_model(conv_model, layer), dense_model))
    models.append(Sequential(to_skip_connection_model(conv_model), dense_model))

    return models
