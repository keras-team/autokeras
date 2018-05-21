from copy import deepcopy
from unittest.mock import patch

from autokeras.search import *
import numpy as np

from tests.common import clean_dir

default_test_path = 'tests/resources/temp'


def mock_train(**kwargs):
    return 1, 0


@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_bayesian_searcher(_):
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2, 3)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1, 3)

    clean_dir(default_test_path)
    generator = BayesianSearcher(3, (28, 28, 1), verbose=False, path=default_test_path)
    for _ in range(2):
        generator.search(x_train, y_train, x_test, y_test)
    clean_dir(default_test_path)
    assert len(generator.history) == len(generator.history_configs)


def test_search_tree():
    tree = SearchTree()
    tree.add_child(-1, 0)
    tree.add_child(0, 1)
    tree.add_child(0, 2)
    assert len(tree.adj_list) == 3
