from tensorflow import keras

from autokeras.tuner import *
from autokeras.hypermodel import HyperModel
from autokeras.hyperparameters import HyperParameters


class MyHyperModel(HyperModel):
    def __init__(self, tune, **kwargs):
        super().__init__(**kwargs)
        self.count = 0
        self.tune = tune

    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))
        for i in range(hp.Choice('num_layers', [2, 4, 6], default=7)):
            model.add(keras.layers.Dense(units=hp.Range('units_' + str(i), 32, 512,
                                                        default=31),
                                         activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        if self.count > 0 and self.tune:
            assert hp.Choice('num_layers', [2, 4, 6], default=7) != 7
        else:
            assert hp.Choice('num_layers', [2, 4, 6], default=7) == 7
        assert hp.Range('units_0', 32, 512, default=31) in [29, 28]
        assert hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]) == 3
        self.count += 1
        return model


def test_reparameterize_and_tune_rest():
    x = np.random.rand(10, 28, 28)
    y = np.random.rand(10, 1)
    val_x = np.random.rand(10, 28, 28)
    val_y = np.random.rand(10, 1)

    hp = HyperParameters()
    hp.Choice('units_0', [29, 28], default=31)

    tuner = SequentialRandomSearch(
        MyHyperModel(True),
        reparameterization=hp,
        tune_rest=True,
        static_values={'learning_rate': 3},
        objective='val_accuracy')

    tuner.search(trials=2,
                 x=x,
                 y=y,
                 epochs=1,
                 validation_data=(val_x, val_y))


def test_reparameterize_and_not_tune_rest():
    x = np.random.rand(10, 28, 28)
    y = np.random.rand(10, 1)
    val_x = np.random.rand(10, 28, 28)
    val_y = np.random.rand(10, 1)

    hp = HyperParameters()
    hp.Choice('units_0', [29, 28], default=31)

    tuner = SequentialRandomSearch(
        MyHyperModel(False),
        reparameterization=hp,
        tune_rest=False,
        static_values={'learning_rate': 3},
        objective='val_accuracy')

    tuner.search(trials=2,
                 x=x,
                 y=y,
                 epochs=1,
                 validation_data=(val_x, val_y))
