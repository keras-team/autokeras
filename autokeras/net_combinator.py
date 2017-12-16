from autokeras.constant import CONV_FUNC_LIST


def layer_num(model, type_list):
    return len(list(filter(lambda layer: isinstance(layer, type_list), model.layers)))


def combine(net1, net2):
    n_conv1 = layer_num(net1, CONV_FUNC_LIST)
    pass
