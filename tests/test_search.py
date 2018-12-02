from unittest.mock import patch

from autokeras.bayesian import edit_distance
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.search import *
from autokeras.nn.generator import CnnGenerator, MlpGenerator, ResNetGenerator
from tests.common import clean_dir, MockProcess, get_classification_data_loaders, get_classification_data_loaders_mlp, \
    simple_transform, TEST_TEMP_DIR, simple_transform_mlp, mock_train, mock_out_of_memory_train


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_bayesian_searcher(_, _1, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    searcher = Searcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
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
    generator = Searcher(3, (28,), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
                         loss=classification_loss, generators=[MlpGenerator, MlpGenerator])
    Constant.N_NEIGHBOURS = 1
    Constant.T_MIN = 0.8
    for _ in range(2):
        generator.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(generator.history) == 2


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_max_acq(_, _2):
    train_data, test_data = get_classification_data_loaders()
    clean_dir(TEST_TEMP_DIR)
    Constant.N_NEIGHBOURS = 2
    Constant.SEARCH_MAX_ITER = 0
    Constant.T_MIN = 0.8
    Constant.BETA = 1
    generator = Searcher(3, (28, 28, 3), verbose=False, path=TEST_TEMP_DIR, metric=Accuracy,
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
    generator = Searcher(3, (28, 28, 3), verbose=True, path=TEST_TEMP_DIR, metric=Accuracy,
                         loss=classification_loss, generators=[CnnGenerator, ResNetGenerator])
    for _ in range(3):
        generator.search(train_data, test_data)
    clean_dir(TEST_TEMP_DIR)
    assert len(generator.history) == 0
