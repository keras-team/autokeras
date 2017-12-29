from keras import Input
from keras.engine import Model
from keras.layers import Flatten, Conv2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta

from autokeras.layer_transformer import *
from autokeras.utils import get_int_tuple, copy_layer
from tests.common import get_conv_model, get_conv_data


def test_deeper_conv_block():
    model = get_conv_model()
    layers = deeper_conv_block(model.layers[1], 3)
    output_tensor = layers[0](model.outputs[0])
    output_tensor = layers[1](output_tensor)
    output_tensor = layers[2](output_tensor)
    new_model = Model(inputs=model.inputs, outputs=output_tensor)
    input_data = get_conv_data()
    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()
    assert np.sum(output1 - output2) < 0.2


def test_dense_to_deeper_layer():
    a = Dense(36, input_shape=(15,), activation='relu')
    model = Sequential([a])
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    a2 = dense_to_deeper_layer(a)
    model2 = Sequential([a, a2])
    model2.compile(loss=categorical_crossentropy,
                   optimizer=Adadelta(),
                   metrics=['accuracy'])
    random_input = np.random.rand(1, 15)
    output1 = model.predict_on_batch(random_input)
    output2 = model2.predict_on_batch(random_input)
    assert np.array_equal(output1, output2)


def test_dense_to_wider_layer():
    a = Dense(20, input_shape=(10,), activation='relu')
    b = Dense(5, activation='relu')
    model = Sequential([a, b])
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    a2, b2 = dense_to_wider_layer(a, b, 5)
    assert a2.units == 25
    model2 = Sequential([a2, b2])
    model2.compile(loss=categorical_crossentropy,
                   optimizer=Adadelta(),
                   metrics=['accuracy'])
    random_input = np.random.rand(1, 10)
    output1 = model.predict_on_batch(random_input)
    output2 = model2.predict_on_batch(random_input)
    assert np.sum(output1.flatten() - output2.flatten()) < 1e-4


def test_conv_to_wider_layer():
    model = get_conv_model()
    conv1 = model.layers[1]
    conv2 = model.layers[4]
    bn1 = model.layers[2]
    new_conv1, [new_conv2], [new_bn1] = conv_to_wider_layer(conv1, [conv2], [bn1], 3)

    new_input = Input(shape=get_int_tuple(model.inputs[0].shape[1:]))
    temp_tensor = new_conv1(new_input)
    temp_tensor = new_bn1(temp_tensor)
    temp_tensor = Activation('relu')(temp_tensor)
    temp_tensor = new_conv2(temp_tensor)
    temp_tensor = copy_layer(model.layers[5])(temp_tensor)
    temp_tensor = Activation('relu')(temp_tensor)
    model2 = Model(inputs=new_input, outputs=temp_tensor)

    random_input = get_conv_data()
    output1 = model.predict_on_batch(random_input)
    output2 = model2.predict_on_batch(random_input)
    assert np.sum(output1.flatten() - output2.flatten()) < 1e-4


def test_conv_dense_to_wider_layer():
    a = Conv2D(20, kernel_size=(1, 1),
               activation='relu',
               input_shape=(28, 28, 1),
               padding='same')
    b = Dense(5, activation='relu')
    model = Sequential([a, Flatten(), b])
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    a2, b2 = conv_dense_to_wider_layer(a, b, 5)
    model2 = Sequential([a2, Flatten(), b2])
    model2.compile(loss=categorical_crossentropy,
                   optimizer=Adadelta(),
                   metrics=['accuracy'])
    random_input = np.random.rand(1, 28, 28, 1)
    output1 = model.predict_on_batch(random_input)
    output2 = model2.predict_on_batch(random_input)
    assert np.sum(output1.flatten() - output2.flatten()) < 1e-4
