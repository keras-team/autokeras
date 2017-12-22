from unittest.mock import patch

from autokeras.search import *
from autokeras import constant
import numpy as np


def simple_transform(_):
    generator = RandomConvClassifierGenerator(input_shape=(28, 28, 1), n_classes=3)
    return [generator.generate(), generator.generate()]


@patch('autokeras.search.transform', side_effect=simple_transform)
def test_hill_climbing_classifier_searcher(_):
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2, 3)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1, 3)

    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 1
    generator = HillClimbingSearcher(3, (28, 28, 1), verbose=False, path=constant.DEFAULT_SAVE_PATH)
    generator.search(x_train, y_train, x_test, y_test)


def test_random_searcher():
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2, 3)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1, 3)

    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 1
    generator = RandomSearcher(3, (28, 28, 1), verbose=False, path=constant.DEFAULT_SAVE_PATH)
    generator.search(x_train, y_train, x_test, y_test)

