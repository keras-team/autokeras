import os

from unittest.mock import patch

import pytest

from autokeras.constant import Constant
from autokeras.text.text_supervised import *
from tests.common import clean_dir, MockProcess, simple_transform, TEST_TEMP_DIR


def mock_train(**kwargs):
    str(kwargs)
    return 1, 0


def mock_text_preprocess(x_train):
    return x_train


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_predict(_, _1, _2):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    clean_dir(TEST_TEMP_DIR)
    clf = TextClassifier(path=TEST_TEMP_DIR, verbose=True)
    train_x = np.random.rand(100, 25, 25)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y, )
    results = clf.predict(train_x)
    assert all(map(lambda result: result in train_y, results))
    clean_dir(TEST_TEMP_DIR)


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
def test_timeout(_, _1):
    # Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1000
    Constant.T_MIN = 0.0001
    Constant.DATA_AUGMENTATION = False
    clean_dir(TEST_TEMP_DIR)
    clf = TextClassifier(path=TEST_TEMP_DIR, verbose=False)
    train_x = np.random.rand(100, 25, 25)
    train_y = np.random.randint(0, 5, 100)
    with pytest.raises(TimeoutError):
        clf.fit(train_x, train_y, time_limit=0)
    clean_dir(TEST_TEMP_DIR)


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
def test_final_fit(_, _1, _2, _3):
    Constant.LIMIT_MEMORY = True
    clean_dir(TEST_TEMP_DIR)
    clf = TextClassifier(path=TEST_TEMP_DIR, verbose=False)
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    Constant.N_NEIGHBOURS = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25, 25)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25, 25)
    test_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    clf.final_fit(train_x, train_y, test_x, test_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    clean_dir(TEST_TEMP_DIR)


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
def test_save_continue(_, _1, _2):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25, 25)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25, 25)
    clean_dir(TEST_TEMP_DIR)
    clf = TextClassifier(path=TEST_TEMP_DIR, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y, time_limit=5)
    assert len(clf.cnn.searcher.history) == 1

    Constant.MAX_MODEL_NUM = 2
    clf = TextClassifier(verbose=False, path=TEST_TEMP_DIR, resume=True)
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.cnn.searcher.history) == 2

    Constant.MAX_MODEL_NUM = 1
    clf = TextClassifier(verbose=False, path=TEST_TEMP_DIR, resume=False)
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.cnn.searcher.history) == 1
    clean_dir(TEST_TEMP_DIR)


@patch('autokeras.supervised.rand_temp_folder_generator', return_value=TEST_TEMP_DIR)
def test_init_text_classifier_with_none_path(_):
    clf = TextClassifier()
    assert clf.path == TEST_TEMP_DIR


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
def test_evaluate(_, _1, _2):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25, 25)
    train_y = np.random.randint(0, 5, 100)
    clean_dir(TEST_TEMP_DIR)
    clf = TextClassifier(path=TEST_TEMP_DIR, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y)
    score = clf.evaluate(train_x, train_y)
    clean_dir(TEST_TEMP_DIR)
    assert score <= 1.0


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
def test_fit_predict_regression(_, _1, _2):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    Constant.DATA_AUGMENTATION = False
    path = TEST_TEMP_DIR
    print(os.getcwd())
    # for f in os.listdir(path):
    #     print(f)
    clean_dir(path)
    clf = TextRegressor(path=path, verbose=False)
    train_x = np.random.rand(100, 25, 25)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    results = clf.predict(train_x)
    assert len(results) == len(train_x)
    clean_dir(path)
