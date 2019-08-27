from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import autokeras as ak


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


def test_structured_data_classifier(tmp_dir):
    # generate high_level dataset
    num_data = 500
    # num_train = 200
    num_features = 8
    num_nan = 100
    data = []
    # 12 classes
    career = ['doctor', 'nurse', 'driver', 'chef', 'teacher', 'writer',
              'actress', 'engineer', 'lawyer', 'realtor', 'agent', 'pilot']
    # 15 classes
    states = ['CA', 'FL', 'GA', 'IL', 'MD',
              'MA', 'MI', 'MN', 'NJ', 'NY',
              'NC', 'PA', 'TX', 'UT', 'VA']
    # 13 classes
    years = ['first', 'second', 'third', 'fourth', 'fifth',
             'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
             'eleventh', 'twelfth', 'thirteenth']
    # 10 classes
    color = ['red', 'orange', 'yellow', 'green', 'blue',
             'purple', 'beige', 'pink', 'silver', 'gold']
    # 3 classes
    size = ['S', 'M', 'L']
    boolean = ['True', 'False']
    career_states = []  # 180 classes
    career_years = []  # 156 classes
    career_color = []  # 120 classes
    career_size = []  # 36 classes
    for c in career:
        for s in states:
            career_states.append(c+'_'+s)
        for y in years:
            career_years.append(c+'_'+y)
        for r in color:
            career_color.append(c+'_'+r)
        for g in size:
            career_size.append(c+'_'+g)
    np.random.seed(0)
    col_bool = np.random.choice(boolean, num_data).reshape(num_data, 1)
    col_num_to_cat = np.random.randint(20, 41, size=num_data).reshape(num_data, 1)
    col_float = 100*np.random.random(num_data,).reshape(num_data, 1)
    col_int = np.random.randint(2000, 4000, size=num_data).reshape(num_data, 1)
    col_morethan_32 = np.random.choice(career_size, num_data).reshape(num_data, 1)
    col1_morethan_100 = np.random.choice(career_states,
                                         num_data).reshape(num_data, 1)
    col2_morethan_100 = np.random.choice(career_years,
                                         num_data).reshape(num_data, 1)
    col3_morethan_100 = np.random.choice(career_color,
                                         num_data).reshape(num_data, 1)
    data = np.concatenate((col_bool, col_num_to_cat, col_float, col_int,
                           col_morethan_32, col1_morethan_100, col2_morethan_100,
                           col3_morethan_100), axis=1)
    # generate np.nan data
    for i in range(num_nan):
        row = np.random.randint(0, num_data)
        col = np.random.randint(0, num_features)
        data[row][col] = np.nan
    # x_train, x_test = data[:num_train], data[num_train:]
    x_train = data
    x_test = data
    y = np.random.randint(0, 3, num_data)
    # y_train, _ = y[:num_train], y[num_train:]
    y_train = y
    clf = ak.StructuredDataClassifier(directory=tmp_dir, max_trials=3)
    clf.fit(x_train, y_train, epochs=2, validation_data=(
        x_train, y_train))
    assert clf.predict(x_test).shape == (len(x_train), 1)
