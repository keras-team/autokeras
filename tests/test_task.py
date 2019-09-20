from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import autokeras as ak
from tests.common import column_names_from_csv
from tests.common import column_names_from_numpy
from tests.common import column_types_from_csv
from tests.common import column_types_from_numpy
from tests.common import false_column_types_from_csv
from tests.common import less_column_names_from_csv
from tests.common import partial_column_types_from_csv
from tests.common import structured_data

train_file_path = r'tests/resources/titanic/train.csv'
test_file_path = r'tests/resources/titanic/eval.csv'


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


def test_structured_data_from_numpy_regressor(tmp_dir):
    num_data = 500
    data = structured_data(num_data)
    x_train = data
    y = np.random.rand(num_data, 1)
    y_train = y
    clf = ak.StructuredDataRegressor(directory=tmp_dir, max_trials=1)
    clf.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))


def test_structured_data_from_numpy_classifier(tmp_dir):
    num_data = 500
    data = structured_data(num_data)
    x_train = data
    y = np.random.randint(0, 3, num_data)
    y_train = y
    clf = ak.StructuredDataClassifier(directory=tmp_dir, max_trials=1)
    clf.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))


def test_structured_data_from_numpy_col_name_classifier(tmp_dir):
    num_data = 500
    data = structured_data(num_data)
    x_train = data
    y = np.random.randint(0, 3, num_data)
    y_train = y
    clf = ak.StructuredDataClassifier(
        column_names=column_names_from_numpy,
        directory=tmp_dir,
        max_trials=1)
    clf.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))


def test_structured_data_from_numpy_col_type_classifier(tmp_dir):
    num_data = 500
    data = structured_data(num_data)
    x_train = data
    y = np.random.randint(0, 3, num_data)
    y_train = y
    with pytest.raises(ValueError) as info:
        clf = ak.StructuredDataClassifier(
            column_types=column_types_from_numpy,
            directory=tmp_dir,
            max_trials=1)
        clf.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    assert str(info.value) == 'Column names must be specified.'


def test_structured_data_from_numpy_col_name_type_classifier(tmp_dir):
    num_data = 500
    data = structured_data(num_data)
    x_train = data
    y = np.random.randint(0, 3, num_data)
    y_train = y
    clf = ak.StructuredDataClassifier(
        column_names=column_names_from_numpy,
        column_types=column_types_from_numpy,
        directory=tmp_dir,
        max_trials=1)
    clf.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))


def test_structured_data_classifier_transform_new_data(tmp_dir):
    num_data = 200
    num_train = 100
    data = structured_data(num_data)
    x_train, x_test = data[:num_train], data[num_train:]
    y = np.random.randint(0, 3, num_data)
    y_train, y_test = y[:num_train], y[num_train:]
    clf = ak.StructuredDataClassifier(directory=tmp_dir, max_trials=1)
    clf.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))


def test_structured_data_from_csv_regressor(tmp_dir):
    clf = ak.StructuredDataRegressor(directory=tmp_dir, max_trials=1)
    clf.fit(x=train_file_path, y='fare', epochs=2, validation_data=test_file_path)


def test_structured_data_from_csv_classifier(tmp_dir):
    clf = ak.StructuredDataClassifier(directory=tmp_dir, max_trials=1)
    clf.fit(x=train_file_path, y='survived', epochs=2,
            validation_data=test_file_path)


def test_structured_data_from_csv_col_name_type_classifier(tmp_dir):
    clf = ak.StructuredDataClassifier(
        column_names=column_names_from_csv,
        column_types=column_types_from_csv,
        directory=tmp_dir,
        max_trials=1)
    clf.fit(x=train_file_path, y='survived', epochs=2,
            validation_data=test_file_path)


def test_structured_data_from_csv_col_name_classifier(tmp_dir):
    clf = ak.StructuredDataClassifier(
        column_names=column_names_from_csv,
        directory=tmp_dir,
        max_trials=1)
    clf.fit(x=train_file_path, y='survived', epochs=2,
            validation_data=test_file_path)


def test_structured_data_from_csv_less_col_name_classifier(tmp_dir):
    with pytest.raises(ValueError) as info:
        clf = ak.StructuredDataClassifier(
            column_names=less_column_names_from_csv,
            directory=tmp_dir,
            max_trials=1)
        clf.fit(x=train_file_path, y='survived', epochs=2,
                validation_data=test_file_path)
    assert str(info.value) == 'The length of column_names and data are mismatched.'


def test_structured_data_from_csv_col_type_mismatch_classifier(tmp_dir):
    with pytest.raises(ValueError) as info:
        clf = ak.StructuredDataClassifier(
            column_types=column_types_from_csv,
            directory=tmp_dir,
            max_trials=1)
        clf.fit(x=train_file_path, y='survived', epochs=2,
                validation_data=test_file_path)
    assert str(info.value) == 'Column_names and column_types are mismatched.'


def test_structured_data_from_csv_false_col_type_classifier(tmp_dir):
    with pytest.raises(ValueError) as info:
        clf = ak.StructuredDataClassifier(
            column_types=false_column_types_from_csv,
            directory=tmp_dir,
            max_trials=1)
        clf.fit(x=train_file_path, y='survived', epochs=2,
                validation_data=test_file_path)
    assert str(info.value) == \
        'Column_types should be either "categorical" or "numerical".'


def test_structured_data_from_csv_partial_col_type_classifier(tmp_dir):
    clf = ak.StructuredDataClassifier(
        column_types=partial_column_types_from_csv,
        directory=tmp_dir,
        max_trials=1)
    clf.fit(x=train_file_path, y='survived', epochs=2,
            validation_data=test_file_path)
