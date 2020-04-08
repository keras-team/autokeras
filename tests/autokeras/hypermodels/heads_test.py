import kerastuner
import numpy as np
import autokeras as ak

from autokeras import nodes as input_module
from autokeras.hypermodels import basic
from autokeras import graph as graph_module
from autokeras.hypermodels import heads as head_module
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


def test_segmentation():
    y = np.array(['a', 'a', 'c', 'b'])
    input_shape = (32,)
    head = head_module.SegmentationHead()
    adapter = head.get_adapter()
    adapter.fit_transform(y)
    head.config_from_adapter(adapter)
    hp = kerastuner.HyperParameters()

    head = graph_module.deserialize(graph_module.serialize(head))
    head.build(hp, ak.Input(shape=input_shape).build())

