import functools
import os

import kerastuner
import numpy as np
import pytest
import tensorflow as tf

from autokeras.hypermodel import preprocessor as preprocessor_module
from tests import common


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('preprocessor_test')


def map_func(x, instance=None, dtype=tf.float32):
    return tf.py_function(instance.transform,
                          inp=[x],
                          Tout=(dtype,))


def run_preprocessor(instance, x, y=None, dtype=tf.float32, tmp_dir=None):
    dataset = tf.data.Dataset.zip((x, y))
    instance.build(kerastuner.HyperParameters())
    for temp_x, temp_y in dataset:
        instance.update(temp_x, temp_y)
    instance.finalize()
    instance.set_state(instance.get_state())
    if tmp_dir:
        instance.save(os.path.join(tmp_dir, 'temp'))

    for temp_x, _ in dataset:
        instance.transform(temp_x, True)

    new_dataset = x.map(functools.partial(map_func,
                                          instance=instance,
                                          dtype=dtype))
    for _ in new_dataset:
        pass
    return new_dataset


def test_normalize():
    dataset = common.generate_data(dtype='dataset')
    new_dataset = run_preprocessor(preprocessor_module.Normalization(),
                                   dataset,
                                   common.generate_data(dtype='dataset'),
                                   dtype=tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_sequence():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    new_dataset = run_preprocessor(
        preprocessor_module.TextToIntSequence(),
        dataset,
        common.generate_data(dtype='dataset'),
        tf.int64)
    assert isinstance(new_dataset, tf.data.Dataset)
    for (x,) in new_dataset:
        assert x.shape == (6,)
        break


def test_ngram():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    new_dataset = run_preprocessor(
        preprocessor_module.TextToNgramVector(),
        dataset,
        common.generate_data(dtype='dataset'),
        tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_augment():
    dataset = common.generate_data(dtype='dataset')
    new_dataset = run_preprocessor(
        preprocessor_module.ImageAugmentation(),
        dataset,
        common.generate_data(dtype='dataset'),
        tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_feature_engineering(tmp_dir):
    dataset = common.generate_structured_data(dtype='dataset')
    feature = preprocessor_module.FeatureEngineering()
    feature.column_names = common.COLUMN_NAMES_FROM_NUMPY
    feature.column_types = common.COLUMN_TYPES_FROM_NUMPY
    new_dataset = run_preprocessor(feature,
                                   dataset,
                                   common.generate_data(dtype='dataset'),
                                   tf.float32,
                                   tmp_dir)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_feature_engineering_new_categorical_value():
    dataset = common.generate_structured_data(num_instances=100, dtype='dataset')
    feature = preprocessor_module.FeatureEngineering()
    feature.column_names = common.COLUMN_NAMES_FROM_NUMPY
    feature.column_types = common.COLUMN_TYPES_FROM_NUMPY
    new_dataset = run_preprocessor(feature,
                                   dataset,
                                   common.generate_data(dtype='dataset'),
                                   tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_lgbm_classifier():
    dataset = common.generate_data(100, (32,), dtype='dataset')
    y = common.generate_one_hot_labels(100, num_classes=3, dtype='dataset')
    instance = preprocessor_module.LightGBMBlock(seed=common.SEED)
    instance.lightgbm_block = preprocessor_module.LightGBMClassifier(
        seed=common.SEED)
    instance.lightgbm_block.num_classes = 3
    new_dataset = run_preprocessor(instance,
                                   dataset,
                                   y,
                                   tf.float32)
    for (x,) in new_dataset:
        assert x.shape == (3,)
        break
    assert isinstance(new_dataset, tf.data.Dataset)


def test_lgbm_classifier_two_classes():
    dataset = common.generate_data(11, (32,), dtype='dataset')
    y = tf.data.Dataset.from_tensor_slices(
        np.random.randint(0, 2, 11).reshape(-1, 1))
    instance = preprocessor_module.LightGBMBlock(seed=common.SEED)
    instance.lightgbm_block = preprocessor_module.LightGBMClassifier(
        seed=common.SEED)
    instance.lightgbm_block.num_classes = 11
    new_dataset = run_preprocessor(instance,
                                   dataset,
                                   y,
                                   tf.float32)
    for (x,) in new_dataset:
        assert x.shape == (1,)
        break
    assert isinstance(new_dataset, tf.data.Dataset)


def test_lgbm_regressor():
    dataset = common.generate_data(11, (32,), dtype='dataset')
    y = common.generate_data(11, (1,), dtype='dataset')
    instance = preprocessor_module.LightGBMBlock(seed=common.SEED)
    instance.lightgbm_block = preprocessor_module.LightGBMRegressor(seed=common.SEED)
    new_dataset = run_preprocessor(instance,
                                   dataset,
                                   y,
                                   tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)
