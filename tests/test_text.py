import numpy as np
from autokeras import const
from autokeras.auto.text import *


def test_image_classifier():
    x_train = np.random.rand(100, 32, 32)
    y_train = np.random.randint(0, 10, 100)
    clf = TextClassifier()
    const.Constant.NUM_TRAILS = 2
    clf.fit(x_train, y_train, epochs=2)
    assert clf.predict(x_train).shape == (100,)


def test_image_regressor():
    x_train = np.random.rand(100, 32, 32)
    y_train = np.random.rand(100)
    clf = TextRegressor()
    const.Constant.NUM_TRAILS = 2
    clf.fit(x_train, y_train, epochs=2)
    assert clf.predict(x_train).shape == (100,)
