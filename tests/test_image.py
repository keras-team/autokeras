import numpy as np
from autokeras.auto.image import *


def test_image_classifier():
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.randint(0, 10, 100)
    clf = ImageClassifier()
    clf.fit(x_train, y_train, trials=2, epochs=1)
    assert clf.predict(x_train).shape == (100,)


def test_image_regressor():
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100)
    clf = ImageRegressor()
    clf.fit(x_train, y_train, trials=2, epochs=1)
    assert clf.predict(x_train).shape == (100,)
