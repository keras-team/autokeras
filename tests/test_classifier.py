from unittest.mock import patch

import pytest

from autokeras.classifier import *
from autokeras import constant
from tests.common import clean_dir


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


class MockProcess(object):
    def __init__(self, target, args):
        self.target = target
        self.args = args

    def join(self):
        pass

    def start(self):
        self.target(*self.args)


@patch('multiprocessing.Process', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=lambda: None)
def test_fit_predict(_):
    constant.MAX_ITER_NUM = 2
    constant.MAX_MODEL_NUM = 2
    constant.EPOCHS_EACH = 1
    constant.N_NEIGHBORS = 1
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = ImageClassifier(path=path, verbose=False)
    train_x = np.array([[[1], [2]], [[3], [4]]])
    train_y = np.array(['a', 'b'])
    clf.fit(train_x, train_y)
    results = clf.predict(train_x)
    assert all(map(lambda result: result in np.array(['a', 'b']), results))
    clean_dir(path)


@patch('multiprocessing.Process', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=lambda: None)
def test_timout(_):
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = ImageClassifier(path=path, verbose=False)
    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 1
    constant.EPOCHS_EACH = 1
    constant.N_NEIGHBORS = 1
    train_x = np.random.rand(100, 25, 25, 1)
    test_x = np.random.rand(50, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y, time_limit=5)
    results = clf.predict(test_x)
    assert len(results) == 50
    clean_dir(path)


@patch('multiprocessing.Process', new=MockProcess)
def test_final_fit():
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = ImageClassifier(path=path, verbose=False)
    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 1
    constant.EPOCHS_EACH = 1
    constant.N_NEIGHBORS = 1
    train_x = np.random.rand(100, 25, 1)
    test_x = np.random.rand(100, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    clf.final_fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    clean_dir(path)


@patch('multiprocessing.Process', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=lambda: None)
def test_save_continue(_):
    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 1
    constant.EPOCHS_EACH = 1
    constant.N_NEIGHBORS = 1
    train_x = np.random.rand(100, 25, 1)
    test_x = np.random.rand(100, 25, 1)
    train_y = np.random.randint(0, 5, 100)
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


@patch('multiprocessing.Process', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=lambda: None)
def test_fit_csv_file_1(_):
    constant.MAX_ITER_NUM = 1
    constant.MAX_MODEL_NUM = 1
    constant.EPOCHS_EACH = 1
    constant.N_NEIGHBORS = 1
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

    clf = ImageClassifier(verbose=False, path=os.path.join(path, "temp"), resume=True)
    clf.fit(csv_file_path=os.path.join(path, "images_test/images_name.csv"),
            images_path=os.path.join(path, "images_test/Black_white_images"))
    img_file_name, y_train = read_csv_file(csv_file_path=os.path.join(path, "images_test/images_name.csv"))
    x_test = read_images(img_file_name, images_dir_path=os.path.join(path, "images_test/Black_white_images"))
    results = clf.predict(x_test)
    assert len(clf.load_searcher().history) == 1
    assert len(results) == 5
    clean_dir(os.path.join(path, "temp"))


@patch('multiprocessing.Process', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=lambda: None)
def test_cross_validate(_):
    constant.MAX_ITER_NUM = 2
    constant.MAX_MODEL_NUM = 2
    constant.EPOCHS_EACH = 1
    constant.N_NEIGHBORS = 1
    path = 'tests/resources/temp'
    clean_dir(path)
    clf = ImageClassifier(path=path, verbose=False)
    train_x = np.random.rand(100, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    results = clf.cross_validate(train_x, train_y, 2)
    clean_dir(path)
    assert len(results) == 2
