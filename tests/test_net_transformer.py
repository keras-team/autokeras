import numpy as np
from autokeras.net_transformer import *
from keras.layers import MaxPooling2D, Dropout, Flatten, MaxPooling1D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta


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
    print(models)
    for new_model in models:
        output2 = new_model.predict_on_batch(random_input)
        assert np.sum(output1.flatten() - output2.flatten()) < 1e-4


def test_net_transformer2():
    model = Sequential()
    img_rows = 5
    input_shape = (img_rows, 1)
    model.add(Conv1D(3, kernel_size=(3,),
                     activation='relu',
                     padding='same',
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=(2,)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    random_input = np.random.rand(1, 5, 1)
    output1 = model.predict_on_batch(random_input)
    models = transform(model)
    print(models)
    for new_model in models:
        output2 = new_model.predict_on_batch(random_input)
        assert np.sum(output1.flatten() - output2.flatten()) < 1e-4
