from keras import Input
from keras.engine import Model
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta

from autokeras.layer_transformer import *
from tests.common import get_conv_model, get_conv_data, get_add_skip_model, get_conv_dense_model


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
    a2 = wider_pre_dense(a, 5)
    b2 = wider_next_dense(b, 20, 20, 5)
    assert a2.units == 25
    model2 = Sequential([a2, b2])
    model2.compile(loss=categorical_crossentropy,
                   optimizer=Adadelta(),
                   metrics=['accuracy'])
    random_input = np.random.rand(1, 10)
    output1 = model.predict_on_batch(random_input)
    output2 = model2.predict_on_batch(random_input)
    assert np.array_equal(output1.flatten(), output2.flatten())


def test_wider_bn():
    bn_layer = get_conv_model().layers[2]
    new_bn_layer = wider_bn(bn_layer, 1, 3, 4)
    assert new_bn_layer.get_weights()[0].shape[0] == 7


def test_wider_weighted_add():
    layer = get_add_skip_model().layers[10]
    new_layer = wider_weighted_add(layer, 4)
    assert isinstance(new_layer, WeightedAdd)


def test_wider_next_dense():
    layer = get_conv_dense_model().layers[5]
    new_layer = wider_next_dense(layer, 3, 3, 3)
    assert new_layer.get_weights()[0].shape == (150, 5)


def test_wider_conv():
    model = get_conv_model()

    layer1 = wider_pre_conv(model.layers[1], 3)
    layer2 = wider_bn(model.layers[2], 3, 3, 3)
    layer3 = wider_next_conv(model.layers[4], 3, 3, 3)

    input_tensor = Input(shape=(5, 5, 3))
    output_tensor = layer1(input_tensor)
    output_tensor = layer2(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = layer3(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation('relu')(output_tensor)
    model2 = Model(inputs=input_tensor, outputs=output_tensor)

    random_input = get_conv_data()
    output1 = model.predict_on_batch(random_input)
    output2 = model2.predict_on_batch(random_input)
    assert np.sum(output1.flatten() - output2.flatten()) < 1e-4
