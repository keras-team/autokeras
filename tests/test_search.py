from copy import deepcopy
from unittest.mock import patch

from autokeras.search import *
import numpy as np

from tests.common import clean_dir, MockProcess

default_test_path = 'tests/resources/temp'


def simple_transform(graph):
    graph.to_concat_skip_model(1, 6)
    return [deepcopy(graph)]


def mock_train(**_):
    return 1, 0


@patch('multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_bayesian_searcher(_, _1):
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2, 3)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1, 3)

    clean_dir(default_test_path)
    generator = BayesianSearcher(3, (28, 28, 1), verbose=False, path=default_test_path)
    constant.N_NEIGHBOURS = 1
    constant.T_MIN = 0.8
    for _ in range(2):
        generator.search(x_train, y_train, x_test, y_test)
    clean_dir(default_test_path)
    assert len(generator.history) == 2


def test_search_tree():
    tree = SearchTree()
    tree.add_child(-1, 0)
    tree.add_child(0, 1)
    tree.add_child(0, 2)
    assert len(tree.adj_list) == 3


@patch('multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_export_json(_, _1):
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2, 3)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1, 3)

    clean_dir(default_test_path)
    generator = BayesianSearcher(3, (28, 28, 1), verbose=False, path=default_test_path)
    constant.N_NEIGHBOURS = 1
    constant.T_MIN = 0.8
    for _ in range(3):
        generator.search(x_train, y_train, x_test, y_test)
    file_path = os.path.join(default_test_path, 'test.json')
    generator.export_json(file_path)
    import json
    data = json.load(open(file_path, 'r'))
    assert len(data['networks']) == 3
    assert len(data['tree']['children']) == 2
    clean_dir(default_test_path)
    assert len(generator.history) == 3
