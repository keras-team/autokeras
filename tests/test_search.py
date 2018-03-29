from copy import deepcopy
from unittest.mock import patch

from autokeras.search import *
from autokeras import constant
import numpy as np

from tests.common import clean_dir

default_test_path = 'tests/resources/temp'


def simple_transform(graph):
    graph.to_concat_skip_model(0, 4)
    return [deepcopy(graph)]


@patch('autokeras.search.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=lambda: None)
def test_hill_climbing_searcher(_, _1):
    # def test_hill_climbing_searcher(_):
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2, 3)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1, 3)

    constant.MAX_MODEL_NUM = 3
    constant.N_NEIGHBORS = 2
    clean_dir(default_test_path)
    generator = HillClimbingSearcher(3, (28, 28, 1), verbose=False, path=default_test_path)
    generator.search(x_train, y_train, x_test, y_test)
    generator.search(x_train, y_train, x_test, y_test)
    clean_dir(default_test_path)
    assert len(generator.history) == len(generator.history_configs)


@patch('autokeras.search.ModelTrainer.train_model', side_effect=lambda: None)
def test_random_searcher(_):
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2, 3)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1, 3)

    constant.MAX_MODEL_NUM = 3
    clean_dir(default_test_path)
    generator = RandomSearcher(3, (28, 28, 1), verbose=False, path=default_test_path)
    generator.search(x_train, y_train, x_test, y_test)
    clean_dir(default_test_path)
    assert len(generator.history) == len(generator.history_configs)


@patch('autokeras.search.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=lambda: None)
def test_bayesian_searcher(_, _1):
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2, 3)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1, 3)

    constant.MAX_MODEL_NUM = 3
    constant.ACQ_EXPLOITATION_DEPTH = 1
    clean_dir(default_test_path)
    generator = BayesianSearcher(3, (28, 28, 1), verbose=False, path=default_test_path)
    generator.search(x_train, y_train, x_test, y_test)
    clean_dir(default_test_path)
    assert len(generator.history) == len(generator.history_configs)


def test_search_tree():
    tree = SearchTree()
    tree.add_child(-1, 0)
    tree.add_child(0, 1)
    tree.add_child(0, 2)
    assert len(tree.adj_list) == 3
