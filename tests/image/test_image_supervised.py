from unittest.mock import patch

import pytest

from autokeras.image.image_supervised import *
from tests.common import clean_dir, MockProcess, simple_transform, mock_train, TEST_TEMP_DIR


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


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.backend.torch.model_trainer.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_predict(_, _1):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    Constant.DATA_AUGMENTATION = False

    clf = ImageClassifier(path=TEST_TEMP_DIR, verbose=True)
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    results = clf.predict(train_x)
    assert all(map(lambda result: result in train_y, results))

    clf = ImageClassifier1D(path=TEST_TEMP_DIR, verbose=True)
    train_x = np.random.rand(100, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    results = clf.predict(train_x)
    assert all(map(lambda result: result in train_y, results))

    clf = ImageClassifier3D(path=TEST_TEMP_DIR, verbose=True)
    train_x = np.random.rand(100, 25, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    results = clf.predict(train_x)
    assert all(map(lambda result: result in train_y, results))

    clf = ImageRegressor1D(path=TEST_TEMP_DIR, verbose=True)
    train_x = np.random.rand(100, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    results = clf.predict(train_x)
    assert len(results) == len(train_y)

    clf = ImageRegressor3D(path=TEST_TEMP_DIR, verbose=True)
    train_x = np.random.rand(100, 25, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    results = clf.predict(train_x)
    assert len(results) == len(train_y)

    clean_dir(TEST_TEMP_DIR)


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
def test_timeout(_):
    # Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1000
    Constant.T_MIN = 0.0001
    Constant.DATA_AUGMENTATION = False
    clean_dir(TEST_TEMP_DIR)
    clf = ImageClassifier(path=TEST_TEMP_DIR, verbose=False)
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    with pytest.raises(TimeoutError):
        clf.fit(train_x, train_y, time_limit=0)
    clean_dir(TEST_TEMP_DIR)


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.backend.torch.model_trainer.ModelTrainer.train_model', side_effect=mock_train)
def test_final_fit(_, _2):
    Constant.LIMIT_MEMORY = True
    clean_dir(TEST_TEMP_DIR)
    clf = ImageClassifier(path=TEST_TEMP_DIR, verbose=False)
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
    clean_dir(TEST_TEMP_DIR)


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.backend.torch.model_trainer.ModelTrainer.train_model', side_effect=mock_train)
def test_save_continue(_, _1):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 1
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    test_x = np.random.rand(100, 25, 25, 1)
    clean_dir(TEST_TEMP_DIR)
    clf = ImageClassifier(path=TEST_TEMP_DIR, verbose=False, resume=False)
    clf.n_epochs = 100
    clf.fit(train_x, train_y)
    assert len(clf.cnn.searcher.history) == 1

    Constant.MAX_MODEL_NUM = 2
    clf = ImageClassifier(verbose=False, path=TEST_TEMP_DIR, resume=True)
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.cnn.searcher.history) == 2

    Constant.MAX_MODEL_NUM = 1
    clf = ImageClassifier(verbose=False, path=TEST_TEMP_DIR, resume=False)
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.cnn.searcher.history) == 1
    clean_dir(TEST_TEMP_DIR)


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.bayesian.transform', side_effect=simple_transform)
@patch('autokeras.backend.torch.model_trainer.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_csv_file(_, _1, _2):
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
    assert len(clf.cnn.searcher.history) == 1
    assert len(results) == 5
    clean_dir(os.path.join(path, "temp"))


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.backend.torch.model_trainer.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_predict_regression(_, _1):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    Constant.DATA_AUGMENTATION = False
    clean_dir(TEST_TEMP_DIR)
    clf = ImageRegressor(path=TEST_TEMP_DIR, verbose=False)
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf.fit(train_x, train_y)
    results = clf.predict(train_x)
    assert len(results) == len(train_x)
    clean_dir(TEST_TEMP_DIR)
