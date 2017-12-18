from keras.layers import Dense, Dropout, MaxPooling1D, MaxPooling2D, MaxPooling3D

from autokeras.utils import is_conv_layer


def compare_network(network_a, network_b):
    """
    Same layer size, same layer structure.
    :param network_a:
    :param network_b:
    :return:
    """
    layers_a = network_a.layers
    layers_b = network_b.layers

    if len(layers_a) != len(layers_b):
        return False

    for index in range(len(layers_b)):
        if not isinstance(layers_b[index], type(layers_a[index])):
            return False

        configuration_a = layers_a[index].get_config()
        configuration_b = layers_b[index].get_config()

        if isinstance(layers_b[index], Dense):
            if (configuration_a['units'] != configuration_b['units'] or
                    configuration_a['activation'] != configuration_b['activation']):
                return False

        if isinstance(layers_b[index], Dropout):
            if configuration_a['rate'] != configuration_b['rate']:
                return False

        if isinstance(layers_b[index], (MaxPooling1D, MaxPooling2D, MaxPooling3D)):
            if configuration_a['pool_size'] != configuration_b['pool_size']:
                return False

        if is_conv_layer(layers_b[index]):
            if (configuration_a['filters'] != configuration_b['filters'] or
                    configuration_a['activation'] != configuration_b['activation'] or
                    configuration_a['kernel_size'] != configuration_b['kernel_size']):
                return False

    return True
