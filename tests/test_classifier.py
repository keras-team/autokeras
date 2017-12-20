import pytest

from autokeras.classifier import *
from autokeras import constant


def test_train_x_array_exception():
    clf = Classifier()
    with pytest.raises(Exception) as info:
        clf.fit(15, [])
    assert str(info.value) == 'x_train should at least has 2 dimensions.'


def test_xy_dim_exception():
    clf = Classifier()
    with pytest.raises(Exception) as info:
        clf.fit([[1, 2], [3, 4]], [6, 7, 8])
    assert str(info.value) == 'x_train and y_train should have the same number of instances.'


def test_x_float_exception():
    clf = Classifier()
    with pytest.raises(Exception) as info:
        clf.fit([[1, 'abc'], [3, 4]], [7, 8])
    assert str(info.value) == 'x_train should only contain numerical data.'


def test_fit_predict():
    constant.MAX_ITER_NUM = 2
    constant.MAX_MODEL_NUM = 2
    clf = ImageClassifier()
    clf.n_epochs = 100
    clf.fit([[[1], [2]], [[3], [4]]], ['a', 'b'])
    results = clf.predict([[[1], [2]], [[3], [4]]])
    print(results)
    assert all(map(lambda result: result in np.array(['a', 'b']), results))


def test_fit_predict2():
    constant.MAX_ITER_NUM = 2
    constant.MAX_MODEL_NUM = 2
    train_x = np.random.rand(100, 25, 1)
    test_x = np.random.rand(100, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf = ImageClassifier()
    clf.n_epochs = 100
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100


def test_save_continue():
    constant.MAX_ITER_NUM = 2
    constant.MAX_MODEL_NUM = 2
    train_x = np.random.rand(100, 25, 1)
    test_x = np.random.rand(100, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    clf = ImageClassifier(path='tests/resources/temp')
    clf.n_epochs = 100
    clf.fit(train_x, train_y)

    constant.MAX_MODEL_NUM = 4
    clf = load_from_path(path='tests/resources/temp')
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
    assert len(clf.history) == 4
