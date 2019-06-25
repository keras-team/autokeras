import pytest
import numpy as np
from autokeras.auto.image import *


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_image')


def test_image_classifier(tmp_dir):
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.randint(0, 10, 100)
    clf = ImageClassifier()
    const.Constant.NUM_TRAILS = 2
    clf.fit(x_train, y_train, epochs=2)
    assert clf.predict(x_train).shape == (100,)


def test_image_regressor(tmp_dir):
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100)
    clf = ImageRegressor()
    const.Constant.NUM_TRAILS = 2
    clf.fit(x_train, y_train, epochs=2)
    assert clf.predict(x_train).shape == (100,)
