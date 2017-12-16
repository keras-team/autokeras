from keras.layers import Conv1D, Conv2D, Conv3D


def get_conv_layer_func(n_dim):
    conv_layer_functions = [Conv1D, Conv2D, Conv3D]
    if n_dim > 4:
        raise ValueError('The input dimension is too high.')
    if n_dim < 2:
        raise ValueError('The input dimension is too low.')
    return conv_layer_functions[n_dim - 2]

