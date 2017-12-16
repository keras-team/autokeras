from keras.layers import Dense

from autokeras.constant import CONV_FUNC_LIST


def layer_num(model, type_list):
    return len(list(filter(lambda layer: isinstance(layer, type_list), model.layers)))


def add_dense(net, n_add):
    return None


def add_conv(net, n_add):
    return None


def combine(net1, net2):
    n_conv1 = layer_num(net1, CONV_FUNC_LIST)
    n_conv2 = layer_num(net2, CONV_FUNC_LIST)
    n_dense1 = layer_num(net1, [Dense])
    n_dense2 = layer_num(net2, [Dense])

    if n_conv1 > n_conv2:
        net2 = add_conv(net2, n_dense1 - n_dense2)
    elif n_conv2 > n_conv1:
        net1 = add_dense(net2, n_dense2 - n_dense1)

    if n_dense1 > n_dense2:
        net2 = add_dense(net2, n_dense1 - n_dense2)
    elif n_dense2 > n_dense1:
        net1 = add_dense(net2, n_dense2 - n_dense1)

