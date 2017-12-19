import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from autokeras.constant import CONV_FUNC_LIST, WEIGHTED_LAYER_FUNC_LIST
from autokeras.net_transformer import copy_layer, to_deeper_model
from autokeras.utils import is_conv_layer


def layer_num(model, type_list):
    return len(list(filter(lambda layer: isinstance(layer, tuple(type_list)), model.layers)))


def add_dense(net, n_add):
    level = None
    for index, layer in enumerate(net.layers):
        if isinstance(layer, Dense):
            level = index
            break
    for i in range(n_add):
        net = to_deeper_model(net, level)
    return net


def add_conv(net, n_add):
    level = None
    for index, layer in enumerate(net.layers):
        if is_conv_layer(layer):
            level = index
            break
    for i in range(n_add):
        net = to_deeper_model(net, level)
    return net


def pad_filter(weight, old_size, new_size):
    pad_width = tuple(map(lambda x: (int(x), int(x)), np.subtract(np.array(new_size), np.array(old_size)) / 2))
    pad_width += ((0, 0), (0, 0))
    return np.pad(weight,
                  pad_width,
                  'constant',
                  constant_values=0)


def combine_conv_weights(layer1, layer2, filter_size, n_input_channel):
    weight1, bias1 = layer1.get_weights()
    weight2, bias2 = layer2.get_weights()

    n_filters1 = layer1.filters
    n_filters2 = layer2.filters

    old_input_channel1 = weight1.shape[-2]
    old_input_channel2 = weight2.shape[-2]

    pad_weight1 = pad_filter(weight1, layer1.kernel_size, filter_size)
    pad_weight1 = np.concatenate((pad_weight1,
                                  np.zeros(filter_size + (n_input_channel - old_input_channel1, n_filters1))),
                                 axis=-2)
    pad_weight2 = pad_filter(weight2, layer2.kernel_size, filter_size)
    pad_weight2 = np.concatenate((np.zeros(filter_size + (n_input_channel - old_input_channel2, n_filters2)),
                                  pad_weight2),
                                 axis=-2)
    return np.concatenate((pad_weight1, pad_weight2), axis=-1), np.concatenate((bias1, bias2))


class NetCombinator:
    def __init__(self):
        self.input_shape = None

    def get_output_channel(self, layer_list):
        last_conv = None
        for layer in layer_list:
            if is_conv_layer(layer):
                last_conv = layer

        if not last_conv:
            return self.input_shape[-1]

        return last_conv.filters

    def combine_layer(self, layer_list, layer1, layer2):
        if is_conv_layer(layer1):
            filter1 = layer1.kernel_size
            filter2 = layer2.kernel_size
            new_filter = tuple(np.array([filter1, filter2]).max(axis=1))
            layer = layer1.__class__(layer1.filters + layer2.filters,
                                     kernel_size=new_filter,
                                     padding='same',
                                     activation='relu')
            n_input_channel = self.get_output_channel(layer_list)
            layer.build((None,) * len(new_filter) + (n_input_channel,))
            layer.set_weights = combine_conv_weights(layer1, layer2, new_filter, n_input_channel)
            return layer

        if isinstance(layer1, Dense):
            weight1, bias1 = layer1.get_weights()
            weight2, bias2 = layer2.get_weights()

            unit1 = layer1.units
            unit2 = layer2.units

            pre_unit1 = layer1.get_weights()[0].shape[0]
            pre_unit2 = layer2.get_weights()[0].shape[0]

            new_layer = Dense(unit1 + unit2, activation='relu')
            new_layer.build((None, pre_unit1 + pre_unit2))
            new_weight = np.zeros((pre_unit1 + pre_unit2, unit1 + unit2))
            new_weight[:pre_unit1, :unit1] = weight1
            new_weight[pre_unit1:, unit1:] = weight2
            new_bias = np.concatenate((bias1, bias2))

            new_layer.set_weights((new_weight, new_bias))
            return new_layer

        return copy_layer(layer1)

    def combine(self, net1, net2):
        self.input_shape = net1.layers[0].input_shape
        n_conv1 = layer_num(net1, CONV_FUNC_LIST)
        n_conv2 = layer_num(net2, CONV_FUNC_LIST)
        n_dense1 = layer_num(net1, [Dense])
        n_dense2 = layer_num(net2, [Dense])

        if n_conv1 > n_conv2:
            net2 = add_conv(net2, n_conv1 - n_conv2)
        elif n_conv2 > n_conv1:
            net1 = add_conv(net1, n_conv2 - n_conv1)

        if n_dense1 > n_dense2:
            net2 = add_dense(net2, n_dense1 - n_dense2)
        elif n_dense2 > n_dense1:
            net1 = add_dense(net2, n_dense2 - n_dense1)

        p1 = 0
        p2 = 0
        layers1 = net1.layers
        layers2 = net2.layers
        new_layer_list = []
        while p1 < len(layers1) and p2 < len(layers2):
            layer1 = layers1[p1]
            layer2 = layers2[p2]
            if type(layer1) == type(layer2):
                new_layer_list.append(self.combine_layer(new_layer_list, layer1, layer2))
                p1 += 1
                p2 += 1
            elif isinstance(layer1, tuple(WEIGHTED_LAYER_FUNC_LIST)):
                new_layer_list.append(copy_layer(layer2))
                p2 += 1
            elif isinstance(layer2, tuple(WEIGHTED_LAYER_FUNC_LIST)):
                new_layer_list.append(copy_layer(layer1))
                p1 += 1

        new_model = Sequential()
        first_layer_config = new_layer_list[0].get_config()
        first_layer_config['input_shape'] = self.input_shape[1:]
        new_model.add(new_layer_list[0].__class__.from_config(first_layer_config))
        for layer in new_layer_list[1:]:
            new_model.add(layer)

        return new_model


def combine(net1, net2):
    return NetCombinator().combine(net1, net2)
