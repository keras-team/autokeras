from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import autokeras as ak
from .common import structured_data


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_image')


@mock.patch('kerastuner.engine.tuner.Tuner.search',
            side_effect=lambda *args, **kwargs: None)
def test_image_classifier(_, tmp_dir):
    x_train = np.random.randn(100, 28, 28, 3)
    y_train = np.random.randint(0, 10, 100)
    clf = ak.ImageClassifier(directory=tmp_dir, max_trials=2)
    clf.fit(x_train, y_train, epochs=2, validation_split=0.2)


@mock.patch('kerastuner.engine.tuner.Tuner.search',
            side_effect=lambda *args, **kwargs: None)
def test_image_regressor(_, tmp_dir):
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)
    clf = ak.ImageRegressor(directory=tmp_dir, max_trials=2)
    clf.fit(x_train, y_train, epochs=2, validation_split=0.2)


def imdb_raw(num_instances=100):
    index_offset = 3  # word index offset

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=1000,
        index_from=index_offset)
    x_train = x_train[:num_instances]
    y_train = y_train[:num_instances].reshape(-1, 1)
    x_test = x_test[:num_instances]
    y_test = y_test[:num_instances].reshape(-1, 1)

    word_to_id = tf.keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}
    x_train = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_train))
    x_test = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_test))
    x_train = np.array(x_train, dtype=np.str)
    x_test = np.array(x_test, dtype=np.str)
    return (x_train, y_train), (x_test, y_test)


def test_text_classifier(tmp_dir):
    (train_x, train_y), (test_x, test_y) = imdb_raw()
    clf = ak.TextClassifier(directory=tmp_dir, max_trials=2)
    clf.fit(train_x, train_y, epochs=2, validation_split=0.2)
    assert clf.predict(test_x).shape == (len(train_x), 1)


def test_text_regressor(tmp_dir):
    (train_x, train_y), (test_x, test_y) = imdb_raw()
    train_y = np.random.rand(100, 1)
    clf = ak.TextRegressor(directory=tmp_dir, max_trials=2)
    clf.fit(train_x, train_y, epochs=2, validation_split=0.2)
    assert clf.predict(test_x).shape == (len(train_x), 1)


def test_structured_data_classifier(tmp_dir):
    num_data = 500
    data = structured_data(num_data)
    # x_train, x_test = data[:num_train], data[num_train:]
    x_train = data
    # x_test = data
    y = np.random.randint(0, 3, num_data)
    # y_train, _ = y[:num_train], y[num_train:]
    y_train = y
    clf = ak.StructuredDataClassifier(directory=tmp_dir, max_trials=1)
    clf.fit(x_train, y_train, epochs=2, validation_data=(
        x_train, y_train))


def test_structured_data_regressor(tmp_dir):
    num_data = 500
    data = structured_data(num_data)
    # x_train, x_test = data[:num_train], data[num_train:]
    x_train = data
    # x_test = data
    y = np.random.rand(num_data, 1)
    # y_train, _ = y[:num_train], y[num_train:]
    y_train = y
    clf = ak.StructuredDataRegressor(directory=tmp_dir, max_trials=2)
    clf.fit(x_train, y_train, epochs=2, validation_data=(
        x_train, y_train))


def test_structured_data_classifier_transform_new_data(tmp_dir):
    num_data = 200
    num_train = 100
    data = structured_data(num_data)
    x_train, x_test = data[:num_train], data[num_train:]
    y = np.random.randint(0, 3, num_data)
    y_train, y_test = y[:num_train], y[num_train:]
    clf = ak.StructuredDataClassifier(directory=tmp_dir, max_trials=2)
    clf.fit(x_train, y_train, epochs=2, validation_data=(
        x_test, y_test))
