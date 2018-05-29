from copy import deepcopy
from unittest.mock import patch

import pytest

from autokeras.classifier import *
from autokeras import constant
from tests.common import clean_dir, MockProcess


def mock_train(**kwargs):
    return 1, 0


def test_train_x_array_exception():
    clf = ImageClassifier()
    with pytest.raises(Exception) as info:
        clf.fit(15, [])
    assert str(info.value) == 'x_train should at least has 2 dimensions.'


def test_xy_dim_exception():
    clf = ImageClassifier()
    with pytest.raises(Exception) as info:
        clf.fit([[1, 2], [3, 4]], [6, 7, 8])
    assert str(info.value) == 'x_train and y_train should have the same number of instances.'


def test_x_float_exception():
    clf = ImageClassifier()
    with pytest.raises(Exception) as info:
        clf.fit([[1, 'abc'], [3, 4]], [7, 8])
    assert str(info.value) == 'x_train should only contain numerical data.'


def simple_transform(graph):
    return [deepcopy(graph), deepcopy(graph)]


@patch('multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_predict(_, _1):
    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 4
    constant.SEARCH_MAX_ITER = 1
    constant.DATA_AUGMENTATION = False
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = ImageClassifier(path=path, verbose=False)
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    results = clf.predict(train_x)
    assert all(map(lambda result: result in train_y, results))
    clean_dir(path)


@patch('multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_timout(_, _1):
    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 4
    constant.SEARCH_MAX_ITER = 1
    constant.DATA_AUGMENTATION = False
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = ImageClassifier(path=path, verbose=False)
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y, time_limit=1)
    clean_dir(path)


@patch('multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_final_fit(_, _1):
    constant.LIMIT_MEMORY = True
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = ImageClassifier(path=path, verbose=False)
    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 1
    constant.SEARCH_MAX_ITER = 1
    constant.N_NEIGHBOURS = 1
    constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25, 25, 1)
    test_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    clf.final_fit(train_x, train_y, test_x, test_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    clean_dir(path)


@patch('multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_save_continue(_, _1):
    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 1
    constant.SEARCH_MAX_ITER = 1
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25, 25, 1)
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = ImageClassifier(path=path, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y)
    assert len(clf.load_searcher().history) == 1

    constant.MAX_MODEL_NUM = 2
    clf = ImageClassifier(verbose=False, path=path, resume=True)
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.load_searcher().history) == 2

    constant.MAX_MODEL_NUM = 1
    clf = ImageClassifier(verbose=False, path=path, resume=False)
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.load_searcher().history) == 1
    clean_dir(path)


@patch('multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_csv_file(_, _1):
    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 1
    constant.SEARCH_MAX_ITER = 1
    path = 'tests/resources'
    clf = ImageClassifier(verbose=False, path=os.path.join(path, "temp"), resume=False)
    clf.fit(csv_file_path=os.path.join(path, "images_test/images_name.csv"),
            images_path=os.path.join(path, "images_test/Color_images"))
    img_file_name, y_train = read_csv_file(csv_file_path=os.path.join(path, "images_test/images_name.csv"))
    x_test = read_images(img_file_name, images_dir_path=os.path.join(path, "images_test/Color_images"))
    results = clf.predict(x_test)
    assert len(clf.load_searcher().history) == 1
    assert len(results) == 5
    clean_dir(os.path.join(path, "temp"))


@patch('multiprocessing.Process', new=MockProcess)
@patch('multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.transform', side_effect=simple_transform)
def test_cross_validate(_):
    constant.MAX_MODEL_NUM = 2
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = ImageClassifier(path=path, verbose=False, searcher_args={'trainer_args': {'max_iter_num': 0}})
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    results = clf.cross_validate(train_x, train_y, 2, trainer_args={'max_iter_num': 0})
    clean_dir(path)
    assert len(results) == 2
