import numpy as np
from autokeras.net_transformer import *
from keras.layers import MaxPooling2D, Dropout, Flatten, MaxPooling1D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta

from tests.common import get_conv_model, get_conv_data


def test_net_transformer():
    model = Sequential()
    img_rows, img_cols = 5, 5
    input_shape = (img_rows, img_cols, 1)
    model.add(Conv2D(3, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    random_input = np.random.rand(1, 5, 5, 1)
    output1 = model.predict_on_batch(random_input)
    models = transform(model)
    for new_model in models:
        output2 = new_model.predict_on_batch(random_input)
        assert np.sum(output1.flatten() - output2.flatten()) < 1e-4


def test_copy_conv_model():
    model = get_conv_model()
    new_model = copy_conv_model(model)
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(output1 - output2) == 0
