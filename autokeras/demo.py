import tensorflow as tf
from tensorflow import keras

import numpy as np

from autokeras.tuner import SequentialRandomSearch
from autokeras.hypermodel import HyperModel
from autokeras.hyperparameters import HyperParameters


(x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
x = x.astype('float32') / 255.
val_x = val_x.astype('float32') / 255.


"""Basic case:
- We define a `build_model` function
- It returns a compiled model
- It uses hyperparameters defined on the fly
"""


def build_model(hp):
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    for i in range(hp.Range('num_layers', 2, 20)):
        model.add(tf.keras.layers.Dense(units=hp.Range('units_' + str(i), 32, 512, 32),
                               activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


tuner = SequentialRandomSearch(
    build_model,
    objective='val_accuracy')

tuner.search(trials=2,
             x=x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))


"""Case #2:
- We override the loss and metrics
"""

tuner = SequentialRandomSearch(
    build_model,
    objective='val_accuracy',
    loss=keras.losses.SparseCategoricalCrossentropy(name='my_loss'),
    metrics=['accuracy', 'mse'])

tuner.search(trials=2,
             x=x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))


"""Case #3:
- We define a HyperModel subclass
"""


class MyHyperModel(HyperModel):

    def __init__(self, img_size, num_classes):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=self.img_size))
        for i in range(hp.Range('num_layers', 2, 20)):
            model.add(tf.keras.layers.Dense(units=hp.Range('units_' + str(i), 32, 512, 32),
                                   activation='relu'))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model


tuner = SequentialRandomSearch(
    MyHyperModel(img_size=(28, 28), num_classes=10),
    objective='val_accuracy')

tuner.search(trials=2,
             x=x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))

"""Case #4:
- We restrict the search space
- This means that default values are being used for params that are left out
"""

hp = HyperParameters()
hp.Choice('learning_rate', [1e-1, 1e-3])

tuner = SequentialRandomSearch(
    build_model,
    reparameterization=hp,
    tune_rest=False,
    objective='val_accuracy')

tuner.search(trials=2,
             x=x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))


"""Case #5:
- We predefine the search space
- No unregistered parameters are allowed in `build`
"""

hp = HyperParameters()
hp.Choice('learning_rate', [1e-1, 1e-3])
hp.Range('num_layers', 2, 20)


def build_model(hp):
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    for i in range(hp.get('num_layers')):
        model.add(tf.keras.layers.Dense(32,
                               activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


tuner = SequentialRandomSearch(
    build_model,
    reparameterization=hp,
    tune_rest=False,
    allow_new_parameters=False,
    objective='val_accuracy')

tuner.search(trials=2,
             x=x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))
