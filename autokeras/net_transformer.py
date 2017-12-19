from autokeras.constant import WEIGHTED_LAYER_FUNC_LIST
from autokeras.layer_transformer import to_deeper_layer, to_wider_layer

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Conv3D

from autokeras.utils import copy_layer


def get_next_dense_conv(start, layers):
    """
    Start to search next dense or conv.
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


def to_deeper_model(model, level):
    new_deeper_layer = to_deeper_layer(model.layers[level])
    return insert_layer(model, level, new_deeper_layer)


def to_wider_model(model, level):
    next_wider_layer, ind = get_next_dense_conv(level, model.layers)
    new_wider_layer, new_next_wider_layer = to_wider_layer(model.layers[level], next_wider_layer, 1)
    return replace_layers(model, [level, ind], [new_wider_layer, new_next_wider_layer])


def net_transformer(model):
    models = []
    layers = model.layers
    for index in range(len(layers) - 1):
        if isinstance(layers[index], tuple(WEIGHTED_LAYER_FUNC_LIST)):
            models.append(to_deeper_model(model, index))
            models.append(to_wider_model(model, index))
    return models
