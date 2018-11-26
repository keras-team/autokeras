from unittest.mock import patch

from autokeras.image.image_supervised import *
from tests.common import MockProcess, mock_train, TEST_TEMP_DIR


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
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
