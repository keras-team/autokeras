from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import autokeras as ak
from tests.common import structured_data


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_image')


@mock.patch('kerastuner.engine.tuner.Tuner.search',
            side_effect=lambda *args, **kwargs: None)
def test_image_classifier(_, tmp_dir):
    x_train = np.random.rand(100, 28, 28, 3)
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


def test_structured_data_from_numpy_classifier(tmp_dir):
    num_data = 500
    data = structured_data(num_data)
    # x_train, x_test = data[:num_train], data[num_train:]
    x_train = data
    # x_test = data
    y = np.random.randint(0, 3, num_data)
    # y_train, _ = y[:num_train], y[num_train:]
    y_train = y
    # clf = ak.StructuredDataClassifier(directory=tmp_dir, max_trials=1)
    clf = ak.StructuredDataClassifier(
        directory=tmp_dir,
        max_trials=1)
    clf.fit(x_train, y_train, epochs=2, validation_data=(
        x_train, y_train))


def test_structured_data_from_numpy_col_name_classifier(tmp_dir):
    num_data = 500
    data = structured_data(num_data)
    # x_train, x_test = data[:num_train], data[num_train:]
    x_train = data
    # x_test = data
    y = np.random.randint(0, 3, num_data)
    # y_train, _ = y[:num_train], y[num_train:]
    y_train = y
    # clf = ak.StructuredDataClassifier(directory=tmp_dir, max_trials=1)
    clf = ak.StructuredDataClassifier(
        column_names=[
                        'bool_',
                        'num_to_cat_',
                        'float_',
                        'int_',
                        'morethan_32_',
                        'col1_morethan_100_',
                        'col2_morethan_100_',
                        'col3_morethan_100_'],
        directory=tmp_dir,
        max_trials=1)
    clf.fit(x_train, y_train, epochs=2, validation_data=(
        x_train, y_train))


# def test_structured_data_from_numpy_col_type_classifier(tmp_dir):
#     num_data = 500
#     data = structured_data(num_data)
#     # x_train, x_test = data[:num_train], data[num_train:]
#     x_train = data
#     # x_test = data
#     y = np.random.randint(0, 3, num_data)
#     # y_train, _ = y[:num_train], y[num_train:]
#     y_train = y
#     # clf = ak.StructuredDataClassifier(directory=tmp_dir, max_trials=1)
#     clf = ak.StructuredDataClassifier(
#         column_types={
#                         'bool_': 'categorical',
#                         'num_to_cat_': 'categorical',
#                         'float_': 'numerical',
#                         'int_': 'numerical',
#                         'morethan_32_': 'categorical',
#                         'col1_morethan_100_': 'categorical',
#                         'col2_morethan_100_': 'categorical',
#                         'col3_morethan_100_': 'categorical'},
#         directory=tmp_dir,
#         max_trials=2)
#     clf.fit(x_train, y_train, epochs=2, validation_data=(
#         x_train, y_train))


def test_structured_data_from_numpy_col_name_type_classifier(tmp_dir):
    num_data = 500
    data = structured_data(num_data)
    # x_train, x_test = data[:num_train], data[num_train:]
    x_train = data
    # x_test = data
    y = np.random.randint(0, 3, num_data)
    # y_train, _ = y[:num_train], y[num_train:]
    y_train = y
    # clf = ak.StructuredDataClassifier(directory=tmp_dir, max_trials=1)
    clf = ak.StructuredDataClassifier(
        column_names=[
                        'bool_',
                        'num_to_cat_',
                        'float_',
                        'int_',
                        'morethan_32_',
                        'col1_morethan_100_',
                        'col2_morethan_100_',
                        'col3_morethan_100_'],
        column_types={
                        'bool_': 'categorical',
                        'num_to_cat_': 'categorical',
                        'float_': 'numerical',
                        'int_': 'numerical',
                        'morethan_32_': 'categorical',
                        'col1_morethan_100_': 'categorical',
                        'col2_morethan_100_': 'categorical',
                        'col3_morethan_100_': 'categorical'},
        directory=tmp_dir,
        max_trials=2)
    clf.fit(x_train, y_train, epochs=2, validation_data=(
        x_train, y_train))


def test_structured_data_from_numpy_regressor(tmp_dir):
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


def test_structured_data_from_csv_regressor(tmp_dir):
    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

    train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
    clf = ak.StructuredDataRegressor(directory=tmp_dir, max_trials=1)
    clf.fit(x=train_file_path, y='fare', epochs=2,
            validation_data=test_file_path)


def test_structured_data_from_csv_classifier(tmp_dir):
    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

    train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
    clf = ak.StructuredDataClassifier(directory=tmp_dir, max_trials=1)
    clf.fit(x=train_file_path, y='survived', epochs=2,
            validation_data=test_file_path)


def test_structured_data_from_csv_col_name_type_classifier(tmp_dir):
    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

    train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
    clf = ak.StructuredDataClassifier(
        column_names=[
                         'sex_',
                         'age_',
                         'n_siblings_spouses_',
                         'parch_',
                         'fare_',
                         'class_',
                         'deck_',
                         'embark_town_',
                         'alone_'],
        column_types={
                          'sex_': 'categorical',
                          'age_': 'numerical',
                          'n_siblings_spouses_': 'categorical',
                          'parch_': 'categorical',
                          'fare_': 'numerical',
                          'class_': 'categorical',
                          'deck_': 'categorical',
                          'embark_town_': 'categorical',
                          'alone_': 'categorical'},
        directory=tmp_dir,
        max_trials=1)
    clf.fit(x=train_file_path, y='survived', epochs=2,
            validation_data=test_file_path)


def test_structured_data_from_csv_col_name_classifier(tmp_dir):
    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

    train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
    clf = ak.StructuredDataClassifier(
        column_names=[
                         'sex_',
                         'age_',
                         'n_siblings_spouses_',
                         'parch_',
                         'fare_',
                         'class_',
                         'deck_',
                         'embark_town_',
                         'alone_'],
        directory=tmp_dir,
        max_trials=1)
    clf.fit(x=train_file_path, y='survived', epochs=2,
            validation_data=test_file_path)


# def test_structured_data_from_csv_col_type_classifier(tmp_dir):
#     TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
#     TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
#
#     train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
#     test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
#     clf = ak.StructuredDataClassifier(
#         column_types={
#                           'sex_': 'categorical',
#                           'age_': 'numerical',
#                           'n_siblings_spouses_': 'categorical',
#                           'parch_': 'categorical',
#                           'fare_': 'numerical',
#                           'class_': 'categorical',
#                           'deck_': 'categorical',
#                           'embark_town_': 'categorical',
#                           'alone_': 'categorical'},
#         directory=tmp_dir,
#         max_trials=1)
#     clf.fit(x=train_file_path, y='survived', epochs=2,
#             validation_data=test_file_path)
#
