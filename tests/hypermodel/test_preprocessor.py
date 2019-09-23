import functools

import numpy as np
import pytest
import tensorflow as tf

import autokeras as ak
import kerastuner
from autokeras.hypermodel import block
from autokeras.hypermodel import head
from autokeras.hypermodel import preprocessor

from ..common import COLUMN_NAMES_FROM_NUMPY
from ..common import COLUMN_TYPES_FROM_NUMPY
from ..common import structured_data


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_lgbm')


def test_normalize():
    normalize = preprocessor.Normalization()
    x_train = np.random.rand(100, 32, 32, 3)
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    normalize.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        normalize.update(x)
    normalize.finalize()
    normalize.set_config(normalize.get_config())

    weights = normalize.get_weights()
    normalize.clear_weights()
    normalize.set_weights(weights)

    for a in dataset:
        normalize.transform(a)

    def map_func(x):
        return tf.py_function(normalize.transform,
                              inp=[x],
                              Tout=(tf.float64,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
    assert isinstance(new_dataset, tf.data.Dataset)


def test_sequence():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    tokenize = preprocessor.TextToIntSequence()
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    tokenize.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        tokenize.update(x)
    tokenize.finalize()
    tokenize.set_config(tokenize.get_config())

    weights = tokenize.get_weights()
    tokenize.clear_weights()
    tokenize.set_weights(weights)

    for a in dataset:
        tokenize.transform(a)

    def map_func(x):
        return tf.py_function(tokenize.transform,
                              inp=[x],
                              Tout=(tf.int64,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
    assert isinstance(new_dataset, tf.data.Dataset)


def test_ngram():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    tokenize = preprocessor.TextToNgramVector()
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    tokenize.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        tokenize.update(x)
    tokenize.finalize()
    tokenize.set_config(tokenize.get_config())

    weights = tokenize.get_weights()
    tokenize.clear_weights()
    tokenize.set_weights(weights)

    for a in dataset:
        tokenize.transform(a)

    def map_func(x):
        return tf.py_function(tokenize.transform,
                              inp=[x],
                              Tout=(tf.float64,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
    assert isinstance(new_dataset, tf.data.Dataset)


def test_augment():
    raw_images = tf.random.normal([1000, 32, 32, 3], mean=-1, stddev=4)
    augment = preprocessor.ImageAugmentation(seed=5)
    dataset = tf.data.Dataset.from_tensor_slices(raw_images)
    hp = kerastuner.HyperParameters()
    augment.set_hp(hp)
    augment.set_config(augment.get_config())
    for a in dataset:
        augment.transform(a, True)

    def map_func(x):
        return tf.py_function(functools.partial(augment.transform, fit=True),
                              inp=[x],
                              Tout=(tf.float32,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
    assert isinstance(new_dataset, tf.data.Dataset)


def test_feature_engineering():
    data = structured_data()
    dataset = tf.data.Dataset.from_tensor_slices(data)
    feature = preprocessor.FeatureEngineering()
    feature.input_node = ak.StructuredDataInput(column_names=COLUMN_NAMES_FROM_NUMPY,
                                                column_types=COLUMN_TYPES_FROM_NUMPY)
    feature.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        feature.update(x)
    feature.finalize()
    feature.set_config(feature.get_config())
    for a in dataset:
        feature.transform(a)

    def map_func(x):
        return tf.py_function(feature.transform,
                              inp=[x],
                              Tout=(tf.float64,))
    new_dataset = dataset.map(map_func)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_feature_engineering_fix_keyerror():
    data = structured_data(100)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    feature = preprocessor.FeatureEngineering()
    feature.input_node = ak.StructuredDataInput(column_names=COLUMN_NAMES_FROM_NUMPY,
                                                column_types=COLUMN_TYPES_FROM_NUMPY)
    feature.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        feature.update(x)
    feature.finalize()
    feature.set_config(feature.get_config())
    for a in dataset:
        feature.transform(a)

    def map_func(x):
        return tf.py_function(feature.transform,
                              inp=[x],
                              Tout=(tf.float64,))
    new_dataset = dataset.map(map_func)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_lgbm_classifier(tmp_dir):
    x_train = np.random.rand(11, 32)
    y_train = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])

    input_node = ak.Input()
    output_node = input_node
    output_node = preprocessor.LightGBMBlock()(output_node)
    output_node = head.ClassificationHead(loss='categorical_crossentropy',
                                          metrics=['accuracy'])(output_node)

    auto_model = ak.GraphAutoModel(input_node,
                                   output_node,
                                   directory=tmp_dir,
                                   max_trials=1)
    auto_model.fit(x_train, y_train, epochs=1,
                   validation_data=(x_train, y_train))
    result = auto_model.predict(x_train)
    assert result.shape == (11, 10)


def test_lgbm_regressor(tmp_dir):
    x_train = np.random.rand(11, 32)
    y_train = np.array([1.1, 2.1, 4.2, 0.3, 2.4, 8.5, 7.3, 8.4, 9.4, 4.3])
    y_train = y_train.reshape(-1, 1)
    input_node = ak.Input()
    output_node = input_node
    output_node = preprocessor.LightGBMBlock()(output_node)
    output_node = head.RegressionHead(loss='mean_squared_error',
                                      metrics=['mean_squared_error'])(output_node)

    auto_model = ak.GraphAutoModel(input_node,
                                   output_node,
                                   directory=tmp_dir,
                                   max_trials=1)
    auto_model.fit(x_train, y_train, epochs=1,
                   validation_data=(x_train, y_train))
    result = auto_model.predict(x_train)
    assert result.shape == (11, 1)
