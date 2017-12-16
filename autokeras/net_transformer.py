from autokeras.layer_transformer import *
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, Flatten,MaxPooling1D, MaxPooling2D, MaxPooling3D,Conv1D, Conv2D, Conv3D
import keras

def copy_layer(layer):
    new_layer = None
    if isinstance(layer, keras.layers.core.Dense):
        new_layer = Dense.from_config(config=layer.get_config())
    elif isinstance(layer, keras.layers.convolutional.Conv1D):
        new_layer = Conv1D.from_config(config=layer.get_config())
    elif isinstance(layer, keras.layers.convolutional.Conv2D):
        new_layer = Conv2D.from_config(config=layer.get_config())
    elif isinstance(layer, keras.layers.convolutional.Conv3D):
        new_layer = Conv3D.from_config(config=layer.get_config())
    if new_layer == None:
        raise ValueError("There must be a Dense or Convolution Layer")
    return new_layer

##start to search next dense or conv
def get_next_dense_conv(start,layers):
    new_next_wider_layer = None
    for j in range(start + 1, len(layers)):
        if isinstance(layers[j], (
                keras.layers.core.Dense, keras.layers.convolutional.Conv1D,
                keras.layers.convolutional.Conv2D, keras.layers.convolutional.Conv3D)):
            new_next_wider_layer = copy_layer(layers[j])
            break
    if new_next_wider_layer == None:
        raise ValueError("There must be a Corresponding Dense or Convolution Layer")
    return new_next_wider_layer


def net_transfromer(model):
    models = []
    layers = model.layers
    for index in range(0,len(layers)-1):
        config = layers[index].get_config()
        weights = layers[index].get_weights()
        new_deeper_model = Sequential()
        new_wider_model = Sequential()
        new_deeper_layer = None
        new_wider_layer = None
        if isinstance(layers[index], (
        keras.layers.core.Dense, keras.layers.convolutional.Conv1D, keras.layers.convolutional.Conv2D,
        keras.layers.convolutional.Conv3D)):
            new_deeper_layer = copy_layer(layers[index])
            new_wider_layer = copy_layer(layers[index])
        else:
            continue
        new_deeper_layer = to_deeper_layer(new_deeper_layer)
        new_next_wider_layer = get_next_dense_conv(index,layers)
        new_wider_layer = to_wider_layer(new_wider_layer,new_next_wider_layer,1)

        #get deeper_model and wider_model
        for i in range(0,len(layers)):
            if i == index:
                new_deeper_model.add(new_deeper_layer)
                new_deeper_layer.set_weights(weights=weights)
                new_wider_model.add(new_wider_layer)
                new_wider_layer.set_weights(weights=weights)
            else:
                new_deeper_model.add(layers[i])
                new_wider_model.add(layers[i])
        models.append(new_deeper_model)
        models.append(new_wider_model)