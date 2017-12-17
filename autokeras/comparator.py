from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, Flatten,MaxPooling1D, MaxPooling2D, MaxPooling3D,Conv1D, Conv2D, Conv3D

def compare_network(network_a,network_b):
    ##same layer size, same layer structure
    layers_a = network_a.layers
    layers_b = network_b.layers
    if len(layers_a) != len(layers_b):
        return False
    for index in range(0,len(layers_b)):
        if type(layers_b[index]) != type(layers_a[index]):
            return False
        else:
            configuration_a = layers_a[index].get_config()
            configuration_b = layers_b[index].get_config()
            if isinstance(layers_b[index], Dense):
                if configuration_a['units'] != configuration_b['units'] or configuration_a['activation'] != configuration_b['activation']:
                    return False
            elif isinstance(layers_b[index], Dropout):
                if configuration_a['rate'] != configuration_b['rate']:
                    return False
            elif isinstance(layers_b[index],Flatten):
                continue
            elif isinstance(layers_b[index],(MaxPooling1D,MaxPooling2D,MaxPooling3D)):
                if configuration_a['pool_size'] != configuration_b['pool_size']:
                    return False
            elif isinstance(layers_b[index],(Conv1D,Conv2D,Conv3D)):
                if configuration_a['filters'] != configuration_b['filters'] or configuration_a['activation'] != configuration_b['activation'] or configuration_a['kernel_size'] != configuration_b['kernel_size']:
                    return False
    return True