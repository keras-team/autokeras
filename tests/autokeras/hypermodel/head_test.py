import kerastuner
import numpy as np
import pytest
import tensorflow as tf

from autokeras.hypermodel import head as head_module
from autokeras.hypermodel import node as node_module
from tests import common


def test_y_is_pd_series():
    (x, y), (val_x, val_y) = common.dataframe_series()
    head = head_module.ClassificationHead()
    head.fit(y)
    assert isinstance(head.transform(y), tf.data.Dataset)


def test_unsupported_types():
    y = 1
    head = head_module.ClassificationHead(name='a')
    with pytest.raises(TypeError) as info:
        head.check_data_type(y)
    assert 'Expect the target data' in str(info.value)


def test_one_class():
    y = np.array(['a', 'a', 'a'])
    head = head_module.ClassificationHead(name='a')
    with pytest.raises(ValueError) as info:
        head.fit(y)
    assert 'Expect the target data' in str(info.value)


def test_two_classes():
    y = np.array(['a', 'a', 'a', 'b'])
    head = head_module.ClassificationHead(name='a')
    head.fit(y)
    head.output_shape = (1,)
    head.build(kerastuner.HyperParameters(), node_module.Input(shape=(32,)).build())
    assert head.loss == 'binary_crossentropy'


def test_three_classes():
    y = np.array(['a', 'a', 'c', 'b'])
    head = head_module.ClassificationHead(name='a')
    head.fit(y)
    assert head.loss == 'categorical_crossentropy'
