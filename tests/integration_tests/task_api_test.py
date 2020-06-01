import tensorflow as tf

import autokeras as ak
from tests import utils


def test_image_classifier(tmp_path):
    train_x = utils.generate_data(num_instances=320, shape=(32, 32, 3))
    train_y = utils.generate_one_hot_labels(num_instances=320, num_classes=10)
    clf = ak.ImageClassifier(directory=tmp_path, max_trials=2, seed=utils.SEED)
    clf.fit(train_x, train_y, epochs=1, validation_split=0.2)
    keras_model = clf.export_model()
    clf.evaluate(train_x, train_y)
    assert clf.predict(train_x).shape == (len(train_x), 10)
    assert isinstance(keras_model, tf.keras.Model)


def test_image_regressor(tmp_path):
    train_x = utils.generate_data(num_instances=320, shape=(32, 32, 3))
    train_y = utils.generate_data(num_instances=320, shape=(1,))
    clf = ak.ImageRegressor(directory=tmp_path, max_trials=2, seed=utils.SEED)
    clf.fit(train_x, train_y, epochs=1, validation_split=0.2)
    clf.export_model()
    assert clf.predict(train_x).shape == (len(train_x), 1)


def test_text_classifier(tmp_path):
    (train_x, train_y), (test_x, test_y) = utils.imdb_raw()
    clf = ak.TextClassifier(directory=tmp_path, max_trials=2, seed=utils.SEED,
                            metrics=['accuracy'], objective='accuracy')
    clf.fit(train_x, train_y, epochs=2, validation_data=(test_x, test_y))
    clf.export_model()
    assert clf.predict(test_x).shape == (len(test_x), 1)
    assert clf.tuner._get_best_trial_epochs() == 2


def test_text_regressor(tmp_path):
    (train_x, train_y), (test_x, test_y) = utils.imdb_raw()
    train_y = utils.generate_data(num_instances=train_y.shape[0], shape=(1,))
    test_y = utils.generate_data(num_instances=test_y.shape[0], shape=(1,))
    clf = ak.TextRegressor(directory=tmp_path, max_trials=2, seed=utils.SEED)
    clf.fit(train_x, train_y, epochs=1, validation_data=(test_x, test_y))
    clf.export_model()
    assert clf.predict(test_x).shape == (len(test_x), 1)


def test_structured_data_from_numpy_regressor(tmp_path):
    num_data = 500
    num_train = 400
    data = utils.generate_structured_data(num_data)
    x_train, x_test = data[:num_train], data[num_train:]
    y = utils.generate_data(num_instances=num_data, shape=(1,))
    y_train, y_test = y[:num_train], y[num_train:]
    clf = ak.StructuredDataRegressor(directory=tmp_path,
                                     max_trials=2,
                                     seed=utils.SEED)
    clf.fit(x_train, y_train, epochs=11, validation_data=(x_train, y_train))
    clf.export_model()
    assert clf.predict(x_test).shape == (len(y_test), 1)


def test_structured_data_from_numpy_classifier(tmp_path):
    num_data = 500
    num_train = 400
    data = utils.generate_structured_data(num_data)
    x_train, x_test = data[:num_train], data[num_train:]
    y = utils.generate_one_hot_labels(num_instances=num_data, num_classes=3)
    y_train, y_test = y[:num_train], y[num_train:]
    clf = ak.StructuredDataClassifier(directory=tmp_path,
                                      max_trials=1,
                                      seed=utils.SEED)
    clf.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    clf.export_model()
    assert clf.predict(x_test).shape == (len(y_test), 3)


def test_timeseries_forecaster(tmp_path):
    lookback = 2
    predict_from = 1
    predict_until = 10
    train_x = utils.generate_data(num_instances=100, shape=(32,))
    train_y = utils.generate_data(num_instances=80, shape=(1,))
    clf = ak.TimeseriesForecaster(lookback=lookback,
                                  directory=tmp_path,
                                  predict_from=predict_from,
                                  predict_until=predict_until,
                                  max_trials=2,
                                  seed=utils.SEED)
    clf.fit(train_x, train_y, epochs=1, validation_split=0.2)
    keras_model = clf.export_model()
    clf.evaluate(train_x, train_y)
    assert clf.predict(train_x).shape == (predict_until - predict_from + 1, 1)
    assert clf.fit_and_predict(train_x,
                               train_y,
                               epochs=1,
                               validation_split=0.2).shape == (predict_until -
                                                               predict_from + 1, 1)
    assert isinstance(keras_model, tf.keras.Model)
