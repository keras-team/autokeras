from autokeras.layer_transformer import *
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, Flatten,MaxPooling1D, MaxPooling2D, MaxPooling3D,Conv1D, Conv2D, Conv3D

def copy_layer(layer):
    new_layer = None
    new_layer=layer.__class__.from_config(layer.get_config())
    if new_layer == None:
        raise ValueError("There must be a Dense or Convolution Layer")
    return new_layer

##start to search next dense or conv
def get_next_dense_conv(start,layers):
    new_next_wider_layer = None
    ind = None
    for j in range(start + 1, len(layers)):
        if isinstance(layers[j], (
                Dense, Conv1D,Conv2D, Conv3D)):
            new_next_wider_layer = layers[j]
            ind = j
            break
    if new_next_wider_layer == None:
        raise ValueError("There must be a Corresponding Dense or Convolution Layer")
    return new_next_wider_layer,ind



def load_to_model(new_model,model,level,new_layer,new_next_wider_layer = None,ind = None, input_shape=None, deeper=False):
    for i in range(0, len(model.layers)):
        if ind != None and i == ind and new_next_wider_layer != None:
            new_model.add(new_next_wider_layer)
        elif i == level:
            if not deeper:
                if i==0:
                    config = new_layer.get_config()
                    config['batch_input_shape'] = input_shape
                    new_layer = new_layer.__class__.from_config(config)
                new_model.add(new_layer)
            else:
                new_model.add(copy_layer(model.layers[i]))
                new_model.layers[-1].set_weights(model.layers[i].get_weights())
                new_model.add(new_layer)
        else:
            new_model.add(copy_layer(model.layers[i]))
            new_model.layers[-1].set_weights(model.layers[i].get_weights())
        #print(new_model.summary())
    return new_model

def to_deeper_model(model, level, input_shape):
    new_deeper_model = Sequential()
    new_deeper_layer = to_deeper_layer(model.layers[level])
    return load_to_model(new_deeper_model,model,level,new_deeper_layer,input_shape=input_shape,deeper=True)

def to_wider_model(model, level, input_shape):
    new_wider_model = Sequential()
    next_wider_layer, ind = get_next_dense_conv(level, model.layers)
    new_wider_layer,new_next_wider_layer = to_wider_layer(model.layers[level], next_wider_layer, 1)
    return load_to_model(new_wider_model,model,level,new_wider_layer,new_next_wider_layer,ind,input_shape)

def net_transfromer(model):
    models = []
    layers = model.layers
    input_shape = layers[0].get_config()['batch_input_shape']
    for index in range(0,len(layers)-1):
        if isinstance(layers[index], (
        Dense, Conv1D, Conv2D,Conv3D)):
            models.append(to_deeper_model(model,index,input_shape))
            models.append(to_wider_model(model,index,input_shape))
    return models