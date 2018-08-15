from copy import deepcopy
from unittest.mock import patch

import pytest

from autokeras.image_classifier import *
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
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_predict(_):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    Constant.DATA_AUGMENTATION = False
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
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_timout(_):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    Constant.DATA_AUGMENTATION = False
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
    Constant.LIMIT_MEMORY = True
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = ImageClassifier(path=path, verbose=False)
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


@patch('multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_save_continue(_):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25, 25, 1)
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = ImageClassifier(path=path, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y)
    assert len(clf.load_searcher().history) == 1

    Constant.MAX_MODEL_NUM = 2
    clf = ImageClassifier(verbose=False, path=path, resume=True)
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.load_searcher().history) == 2

    Constant.MAX_MODEL_NUM = 1
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
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    path = 'tests/resources'
    clf = ImageClassifier(verbose=False, path=os.path.join(path, "temp"), resume=False)
    x_train, y_train = load_image_dataset(csv_file_path=os.path.join(path, "images_test/images_name.csv"),
                                          images_path=os.path.join(path, "images_test/Color_images"))
    clf.fit(x_train, y_train)
    x_test, y_test = load_image_dataset(csv_file_path=os.path.join(path, "images_test/images_name.csv"),
                                        images_path=os.path.join(path, "images_test/Color_images"))
    results = clf.predict(x_test)
    assert len(clf.load_searcher().history) == 1
    assert len(results) == 5
    clean_dir(os.path.join(path, "temp"))


@patch('autokeras.image_classifier.temp_folder_generator', return_value='dummy_path/')
def test_init_image_classifier_with_none_path(_):
    clf = ImageClassifier()
    assert clf.path == 'dummy_path/'
