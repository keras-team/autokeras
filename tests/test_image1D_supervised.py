from unittest.mock import patch

import pytest

from autokeras.image1D_supervised import *
from tests.common import clean_dir, MockProcess, simple_transform
from autokeras.constant import Constant
import os

def mock_train(**kwargs):
    str(kwargs)
    return 1, 0


def test_train_x_array_exception():
    clf = Image1DClassifier()
    with pytest.raises(Exception) as info:
        clf.fit(15, [])
    assert str(info.value) == 'x_train should have exactly 2 dimensions.'


def test_xy_dim_exception():
    clf = Image1DClassifier()
    with pytest.raises(Exception) as info:
        clf.fit([[1, 2], [3, 4]], [6, 7, 8])
    assert str(info.value) == 'x_train and y_train should have the same number of instances.'


def test_x_float_exception():
    clf = Image1DClassifier()
    with pytest.raises(Exception) as info:
        clf.fit([[1, 'abc'], [3, 4]], [7, 8])
    assert str(info.value) == 'x_train should only contain numerical data.'


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_predict(_):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    Constant.DATA_AUGMENTATION = False
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = Image1DClassifier(path=path, verbose=True)
    train_x = np.random.rand(100, 25)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y, )
    results = clf.predict(train_x)
    assert all(map(lambda result: result in train_y, results))
    clean_dir(path)


@patch('torch.multiprocessing.Pool', new=MockProcess)
def test_timeout():
    # Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1000
    Constant.T_MIN = 0.0001
    Constant.DATA_AUGMENTATION = False
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = Image1DClassifier(path=path, verbose=False)
    train_x = np.random.rand(100, 25)
    train_y = np.random.randint(0, 5, 100)
    with pytest.raises(TimeoutError):
        clf.fit(train_x, train_y, time_limit=1)
    clean_dir(path)


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_timeout_resume(_):
    Constant.MAX_ITER_NUM = 1
    # make it impossible to complete within 10sec
    Constant.MAX_MODEL_NUM = 1000
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25)
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = Image1DClassifier(path=path, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y, 15)
    history_len = len(clf.load_searcher().history)
    assert history_len != 0
    results = clf.predict(test_x)
    assert len(results) == 100

    clf = Image1DClassifier(verbose=False, path=path, resume=True)
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
def test_final_fit(_, _1):
    Constant.LIMIT_MEMORY = True
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = Image1DClassifier(path=path, verbose=False)
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    Constant.N_NEIGHBOURS = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25)
    test_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    clf.final_fit(train_x, train_y, test_x, test_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    clean_dir(path)


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_save_continue(_):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25)
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = Image1DClassifier(path=path, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y)
    assert len(clf.load_searcher().history) == 1

    Constant.MAX_MODEL_NUM = 2
    clf = Image1DClassifier(verbose=False, path=path, resume=True)
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.load_searcher().history) == 2

    Constant.MAX_MODEL_NUM = 1
    clf = Image1DClassifier(verbose=False, path=path, resume=False)
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.load_searcher().history) == 1
    clean_dir(path)


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_csv_file(_, _1):
    pass

@patch('autokeras.image_supervised.temp_folder_generator', return_value='dummy_path/')
def test_init_image_classifier_with_none_path(_):
    clf = Image1DClassifier()
    assert clf.path == 'dummy_path/'


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_predict_regression(_):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    Constant.DATA_AUGMENTATION = False
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = Image1DRegressor(path=path, verbose=False)
    train_x = np.random.rand(100, 25)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y, )
    results = clf.predict(train_x)
    assert len(results) == len(train_x)
    clean_dir(path)


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_export_keras_model(_):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25)
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = Image1DClassifier(path=path, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y)
    score = clf.evaluate(train_x, train_y)
    assert score <= 1.0

    test_x = clf.reshapeTo2D(test_x) #for saved processing.
    train_x_2d = clf.reshapeTo2D(train_x) #for saved processing.

    model_file_name = os.path.join(path, 'test_keras_model.h5')
    clf.export_keras_model(model_file_name)
    from keras.models import load_model
    model = load_model(model_file_name)
    results = model.predict(test_x)
    assert len(results) == len(test_x)
    del model, results, model_file_name

    model_file_name = os.path.join(path, 'test_autokeras_model.pkl')
    clf.export_autokeras_model(model_file_name)
    from autokeras.utils import pickle_from_file
    model = pickle_from_file(model_file_name)
    results = model.predict(test_x)
    assert len(results) == len(test_x)
    score = model.evaluate(train_x_2d, train_y)
    assert score <= 1.0
    clean_dir(path)

    clf = Image1DRegressor(path=path, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y)
    score = clf.evaluate(train_x, train_y)
    assert score >= 0.0

    model_file_name = os.path.join(path, 'test_keras_model.h5')
    clf.export_keras_model(model_file_name)
    from keras.models import load_model
    model = load_model(model_file_name)
    results = model.predict(test_x)
    assert len(results) == len(test_x)
    del model, results, model_file_name

    model_file_name = os.path.join(path, 'test_autokeras_model.pkl')
    clf.export_autokeras_model(model_file_name)
    from autokeras.utils import pickle_from_file
    model = pickle_from_file(model_file_name)
    results = model.predict(test_x)
    assert len(results) == len(test_x)
    score = model.evaluate(train_x_2d, train_y)
    assert score >= 0.0
    clean_dir(path)
