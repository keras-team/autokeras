from unittest.mock import patch

import pytest

from autokeras.text.text_supervised import *
from tests.common import clean_dir, MockProcess, simple_transform


def mock_train(**kwargs):
    str(kwargs)
    return 1, 0


def mock_text_preprocess(x_train, path="dummy_path"):
    return x_train


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_predict(_, _1):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = TextClassifier(path=path, verbose=True)
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y, )
    results = clf.predict(train_x)
    assert all(map(lambda result: result in train_y, results))
    clean_dir(path)


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
def test_timeout(_):
    # Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1000
    Constant.T_MIN = 0.0001
    Constant.DATA_AUGMENTATION = False
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = TextClassifier(path=path, verbose=False)
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    with pytest.raises(TimeoutError):
        clf.fit(train_x, train_y, time_limit=0)
    clean_dir(path)


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_timeout_resume(_, _1):
    Constant.MAX_ITER_NUM = 1
    # make it impossible to complete within 10sec
    Constant.MAX_MODEL_NUM = 1000
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25, 25, 1)
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = TextClassifier(path=path, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y, time_limit=2)
    history_len = len(clf.load_searcher().history)
    assert history_len != 0
    results = clf.predict(test_x)
    assert len(results) == 100

    clf = TextClassifier(path=path, verbose=False, resume=True)
    assert len(clf.load_searcher().history) == history_len
    Constant.MAX_MODEL_NUM = history_len + 1
    clf.fit(train_x, train_y)
    assert len(clf.load_searcher().history) == history_len + 1
    results = clf.predict(test_x)
    assert len(results) == 100
    clean_dir(path)


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
def test_final_fit(_, _1, _2):
    Constant.LIMIT_MEMORY = True
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = TextClassifier(path=path, verbose=False)
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    Constant.N_NEIGHBOURS = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25, 25, 1)
    test_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    clf.final_fit(train_x, train_y, test_x, test_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    clean_dir(path)


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
def test_save_continue(_, _1):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25, 25, 1)
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = TextClassifier(path=path, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y, time_limit=5)
    assert len(clf.load_searcher().history) == 1

    Constant.MAX_MODEL_NUM = 2
    clf = TextClassifier(verbose=False, path=path, resume=True)
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.load_searcher().history) == 2

    Constant.MAX_MODEL_NUM = 1
    clf = TextClassifier(verbose=False, path=path, resume=False)
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.load_searcher().history) == 1
    clean_dir(path)


@patch('autokeras.text.text_supervised.temp_folder_generator', return_value='dummy_path/')
def test_init_image_classifier_with_none_path(_):
    clf = TextClassifier()
    assert clf.path == 'dummy_path/'


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
def test_evaluate(_, _1):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = TextClassifier(path=path, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y)
    score = clf.evaluate(train_x, train_y)
    assert score <= 1.0
