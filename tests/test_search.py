from unittest.mock import patch

from autokeras.bayesian import edit_distance
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.search import *
from autokeras.nn.generator import CnnGenerator, MlpGenerator, ResNetGenerator
from tests.common import clean_dir, MockProcess, get_classification_data_loaders, get_classification_data_loaders_mlp, \
    simple_transform, TEST_TEMP_DIR, simple_transform_mlp, mock_train, mock_out_of_memory_train, \
    mock_exception_handling_train

from nas.greedy import GreedySearcher
from nas.grid import GridSearcher
from nas.random import RandomSearcher


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_bayesian_searcher(_, _1, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = BayesianSearcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                                loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    Constant.N_NEIGHBOURS = 1
    Constant.T_MIN = 0.8
    for _ in range(2):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_exception_handling_train)
def test_exception_handling(_, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    Constant.N_NEIGHBOURS = 2
    Constant.T_MIN = 0.8
    Constant.BETA = 1
    generator = BayesianSearcher(3, (28, 28, 3), verbose=True, path=TEST_TEMP_DIR, metric=Accuracy,
                                 loss=classification_loss, generators=[CnnGenerator, ResNetGenerator])
    for _ in range(3):
        generator.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(generator.history) == 0


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.search.get_system', return_value=Constant.SYS_GOOGLE_COLAB)
def test_bayesian_searcher_sp(_, _1, _2, _3):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = BayesianSearcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                                loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    Constant.N_NEIGHBOURS = 1
    Constant.T_MIN = 0.8
    for _ in range(2):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform_mlp)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_bayesian_searcher_mlp(_, _1, _2):
    train_data, test_data = get_classification_data_loaders_mlp()
    clean_dir(TEST_TEMP_DIR)
    generator = BayesianSearcher(3, (28,), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                                 loss=classification_loss, generators=[MlpGenerator, MlpGenerator])
    Constant.N_NEIGHBOURS = 1
    Constant.T_MIN = 0.8
    for _ in range(2):
        generator.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(generator.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_greedy_searcher(_, _1, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = GreedySearcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                              loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    for _ in range(2):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.search.get_system', return_value=Constant.SYS_GOOGLE_COLAB)
def test_greedy_searcher_sp(_, _1, _2, _3):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = GreedySearcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                              loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    for _ in range(2):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform_mlp)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_greedy_searcher_mlp(_, _1, _2):
    train_data, test_data = get_classification_data_loaders_mlp()
    clean_dir(TEST_TEMP_DIR)
    generator = GreedySearcher(3, (28,), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                               loss=classification_loss, generators=[MlpGenerator, MlpGenerator])
    for _ in range(2):
        generator.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(generator.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_random_searcher(_, _1, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = RandomSearcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                              loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    for _ in range(2):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.search.get_system', return_value=Constant.SYS_GOOGLE_COLAB)
def test_random_searcher_sp(_, _1, _2, _3):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = RandomSearcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                              loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    for _ in range(2):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_grid_searcher(_, _1, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = GridSearcher(3, (28, 28, 3), verbose=True, path=TEST_TEMP_DIR, metric=Accuracy,
                            loss=classification_loss, generators=[CnnGenerator, CnnGenerator])
    Constant.N_NEIGHBOURS = 1
    Constant.T_MIN = 0.8
    print(len(searcher.get_search_dimensions()))
    for _ in range(len(searcher.get_search_dimensions())):
        searcher.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(searcher.history) == len(searcher.search_dimensions)


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_max_acq(_, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    Constant.N_NEIGHBOURS = 2
    Constant.SEARCH_MAX_ITER = 0
    Constant.T_MIN = 0.8
    Constant.BETA = 1
    generator = BayesianSearcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                                 loss=classification_loss, generators=[CnnGenerator, ResNetGenerator])
    for _ in range(3):
        generator.search(train_data, test_data)
    for index1, descriptor1 in enumerate(generator.descriptors):
        for descriptor2 in generator.descriptors[index1 + 1:]:
            assert edit_distance(descriptor1, descriptor2) != 0.0

    clean_dir(TEST_TEMP_DIR)


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_out_of_memory_train)
def test_out_of_memory(_, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    Constant.N_NEIGHBOURS = 2
    Constant.SEARCH_MAX_ITER = 0
    Constant.T_MIN = 0.8
    Constant.BETA = 1
    generator = BayesianSearcher(3, (28, 28, 3), verbose=True, path=TEST_TEMP_DIR, metric=Accuracy,
                                 loss=classification_loss, generators=[CnnGenerator, ResNetGenerator])
    for _ in range(3):
        generator.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(generator.history) == 0
