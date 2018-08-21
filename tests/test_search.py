from unittest.mock import patch

from autokeras.loss_function import classification_loss
from autokeras.metric import Accuracy
from autokeras.search import *

from tests.common import clean_dir, MockProcess, get_classification_data_loaders, get_add_skip_model, \
    get_concat_skip_model, simple_transform

default_test_path = 'tests/resources/temp'


def mock_train(**_):
    return 1, 0


@patch('multiprocessing.Pool', new=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_bayesian_searcher(_, _1):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(default_test_path)
    generator = Searcher(3, (28, 28, 3), verbose=False, path=default_test_path, metric=Accuracy,
                         loss=classification_loss)
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
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_export_json(_, _1):
    train_data, test_data = get_classification_data_loaders()

    clean_dir(default_test_path)
    generator = Searcher(3, (28, 28, 3), verbose=False, path=default_test_path, metric=Accuracy,
                         loss=classification_loss)
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


def test_graph_duplicate():
    assert same_graph(get_add_skip_model().extract_descriptor(), get_add_skip_model().extract_descriptor())
    assert not same_graph(get_concat_skip_model().extract_descriptor(), get_add_skip_model().extract_descriptor())


@patch('multiprocessing.Pool', new=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_max_acq(_, _1):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(default_test_path)
    Constant.N_NEIGHBOURS = 2
    Constant.SEARCH_MAX_ITER = 0
    Constant.T_MIN = 0.8
    Constant.BETA = 1
    generator = Searcher(3, (28, 28, 3), verbose=False, path=default_test_path, metric=Accuracy,
                         loss=classification_loss)
    for _ in range(3):
        generator.search(train_data, test_data)
    for index1, descriptor1 in enumerate(generator.descriptors):
        for descriptor2 in generator.descriptors[index1 + 1:]:
            assert edit_distance(descriptor1, descriptor2, 1) != 0

    clean_dir(default_test_path)
