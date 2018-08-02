from copy import deepcopy
from unittest.mock import patch

from autokeras.search import *

from tests.common import clean_dir, MockProcess, get_processed_data

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
    train_data, test_data = get_processed_data()
    clean_dir(default_test_path)
    generator = BayesianSearcher(3, (28, 28, 3), verbose=False, path=default_test_path)
    Constant.N_NEIGHBOURS = 1
    Constant.T_MIN = 0.8
    for _ in range(2):
        generator.search(train_data, test_data)
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
    train_data, test_data = get_processed_data()

    clean_dir(default_test_path)
    generator = BayesianSearcher(3, (28, 28, 3), verbose=False, path=default_test_path)
    Constant.N_NEIGHBOURS = 1
    Constant.T_MIN = 0.8
    for _ in range(3):
        generator.search(train_data, test_data)
    file_path = os.path.join(default_test_path, 'test.json')
    generator.export_json(file_path)
    import json
    data = json.load(open(file_path, 'r'))
    assert len(data['networks']) == 3
    assert len(data['tree']['children']) == 2
    clean_dir(default_test_path)
    assert len(generator.history) == 3
