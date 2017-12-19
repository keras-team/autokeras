import pytest

from autokeras.classifier import *


def test_train_x_array_exception():
    clf = Classifier()
    with pytest.raises(Exception) as info:
        clf.fit(15, [])
    assert str(info.value) == 'x_train should be a 2d array.'


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
    clf = Classifier()
    clf.n_epochs = 100
    clf.fit([[1, 2], [3, 4]], ['a', 'b'])
    results = clf.predict([[1, 2], [3, 4]])
    assert np.array_equal(results, np.array(['a', 'b']))


def test_fit_predict2():
    train_x = np.random.rand(100, 25)
    test_x = np.random.rand(100, 25)
    train_y = np.random.randint(0, 5, 100)
    clf = Classifier()
    clf.n_epochs = 100
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
    assert len(results) == 100
