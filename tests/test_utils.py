from keras.layers import MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential

from autokeras.utils import *
from autokeras.search import *
import numpy as np


def test_model_rainer():
    model = RandomConvClassifierGenerator(3, (28, 28, 1)).generate()
    ModelTrainer(model, np.random.rand(2, 28, 28, 1), np.random.rand(2, 3), np.random.rand(1, 28, 28, 1),
                 np.random.rand(1, 3), True).train_model()


def test_extract_config1():
    model_a = Sequential()

    model_a.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model_a.add(Conv2D(32, (3, 3), activation='relu'))
    model_a.add(MaxPooling2D(pool_size=(2, 2)))
    model_a.add(Dropout(0.25))

    model_a.add(Flatten())
    model_a.add(Dense(128, activation='relu'))
    model_a.add(Dropout(0.5))
    model_a.add(Dense(10, activation='softmax'))

    model_b = Sequential()

    model_b.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model_b.add(Conv2D(32, (3, 3), activation='relu'))
    model_b.add(MaxPooling2D(pool_size=(2, 2)))
    model_b.add(Dropout(0.25))

    model_b.add(Flatten())
    model_b.add(Dense(128, activation='relu'))
    model_b.add(Dropout(0.5))
    model_b.add(Dense(10, activation='softmax'))

    assert extract_config(model_a) == extract_config(model_b)


def test_extract_config2():
    model_a = Sequential()

    model_a.add(Conv2D(32, (3, 3), activation='softmax', input_shape=(28, 28, 1)))
    model_a.add(Conv2D(32, (3, 3), activation='relu'))
    model_a.add(MaxPooling2D(pool_size=(2, 2)))
    model_a.add(Dropout(0.25))

    model_a.add(Flatten())
    model_a.add(Dense(128, activation='relu'))
    model_a.add(Dropout(0.5))
    model_a.add(Dense(10, activation='softmax'))

    model_b = Sequential()

    model_b.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model_b.add(Conv2D(32, (3, 3), activation='relu'))
    model_b.add(MaxPooling2D(pool_size=(2, 2)))
    model_b.add(Dropout(0.25))

    model_b.add(Flatten())
    model_b.add(Dense(128, activation='relu'))
    model_b.add(Dropout(0.5))
    model_b.add(Dense(10, activation='softmax'))
    print(extract_config(model_a))
    assert extract_config(model_a) != extract_config(model_b)
