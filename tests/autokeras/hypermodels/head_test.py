import kerastuner
import numpy as np
import pytest
import tensorflow as tf

from autokeras.hypermodels import head as head_module
from autokeras.hypermodels import input_node as input_module
from tests import utils


def test_two_classes():
    y = np.array(['a', 'a', 'a', 'b'])
    head = head_module.ClassificationHead(name='a')
    adapter = head.get_adapter()
    adapter.fit_transform(y)
    head.config_from_adapter(adapter)
    head.output_shape = (1,)
    head.build(kerastuner.HyperParameters(), input_module.Input(shape=(32,)).build())
    assert head.loss == 'binary_crossentropy'


def test_three_classes():
    y = np.array(['a', 'a', 'c', 'b'])
    head = head_module.ClassificationHead(name='a')
    adapter = head.get_adapter()
    adapter.fit_transform(y)
    head.config_from_adapter(adapter)
    assert head.loss == 'categorical_crossentropy'
