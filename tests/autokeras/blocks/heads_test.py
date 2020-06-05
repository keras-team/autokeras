import kerastuner
import numpy as np
import tensorflow as tf

import autokeras as ak
from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.blocks import heads as head_module


def test_two_classes():
    y = np.array(['a', 'a', 'a', 'b'])
    head = head_module.ClassificationHead(name='a')
    adapter = head.get_adapter()
    adapter.fit_transform(y)
    head.config_from_adapter(adapter)
    head.output_shape = (1,)
    head.build(kerastuner.HyperParameters(), input_module.Input(shape=(32,)).build())
    assert head.loss.name == 'binary_crossentropy'


def test_three_classes():
    y = np.array(['a', 'a', 'c', 'b'])
    head = head_module.ClassificationHead(name='a')
    adapter = head.get_adapter()
    adapter.fit_transform(y)
    head.config_from_adapter(adapter)
    assert head.loss.name == 'categorical_crossentropy'


def test_multi_label_loss():
    head = head_module.ClassificationHead(name='a', multi_label=True, num_classes=8)
    head.output_shape = (8,)
    input_node = tf.keras.Input(shape=(5,))
    output_node = head.build(kerastuner.HyperParameters(), input_node)
    model = tf.keras.Model(input_node, output_node)
    assert model.layers[-1].activation.__name__ == 'sigmoid'
    assert head.loss.name == 'binary_crossentropy'


def test_segmentation():
    y = np.array(['a', 'a', 'c', 'b'])
    head = head_module.SegmentationHead(name='a')
    adapter = head.get_adapter()
    adapter.fit_transform(y)
    head.config_from_adapter(adapter)
    input_shape = (64, 64, 21)
    hp = kerastuner.HyperParameters()
    head = blocks.deserialize(blocks.serialize(head))
    head.build(hp, ak.Input(shape=input_shape).build())
