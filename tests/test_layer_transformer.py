from keras.layers import MaxPooling2D, Dropout, Flatten
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta

from autokeras.layer_transformer import *


def test_dense_to_deeper_layer():
    a = Dense(36, input_shape=(15,), activation='relu')
    model = Sequential([a])
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    a2 = to_deeper_layer(a)
    model2 = Sequential([a, a2])
    model2.compile(loss=categorical_crossentropy,
                   optimizer=Adadelta(),
                   metrics=['accuracy'])
    random_input = np.random.rand(1, 15)
    output1 = model.predict_on_batch(random_input)
    output2 = model2.predict_on_batch(random_input)
    assert np.array_equal(output1, output2)


def test_conv_to_deeper_layer():
    a = Conv2D(32, kernel_size=(3, 3),
               activation='relu',
               input_shape=(28, 28, 1),
               padding='same')
    model = Sequential([a])
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    a2 = to_deeper_layer(a)
    model2 = Sequential([a, a2])
    model2.compile(loss=categorical_crossentropy,
                   optimizer=Adadelta(),
                   metrics=['accuracy'])
    random_input = np.random.rand(1, 28, 28, 1)
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
    a2, b2 = to_wider_layer(a, b, 5)
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
    a = Conv2D(20, kernel_size=(1, 1),
               activation='relu',
               input_shape=(28, 28, 1),
               padding='same')
    b = Conv2D(30, kernel_size=(1, 1),
               activation='relu',
               padding='same')
    model = Sequential([a, b])
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    a2, b2 = to_wider_layer(a, b, 5)
    model2 = Sequential([a2, b2])
    model2.compile(loss=categorical_crossentropy,
                   optimizer=Adadelta(),
                   metrics=['accuracy'])
    random_input = np.random.rand(1, 28, 28, 1)
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
    a2, b2 = to_wider_layer(a, b, 5)
    model2 = Sequential([a2, Flatten(), b2])
    model2.compile(loss=categorical_crossentropy,
                   optimizer=Adadelta(),
                   metrics=['accuracy'])
    random_input = np.random.rand(1, 28, 28, 1)
    output1 = model.predict_on_batch(random_input)
    output2 = model2.predict_on_batch(random_input)
    assert np.sum(output1.flatten() - output2.flatten()) < 1e-4
