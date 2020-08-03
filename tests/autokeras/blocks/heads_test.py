# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kerastuner
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

import autokeras as ak
from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.blocks import heads as head_module


def test_two_classes():
    y = np.array(["a", "a", "a", "b"])
    head = head_module.ClassificationHead(name="a")
    adapter = head.get_adapter()
    adapter.fit_transform(y)
    head.config_from_adapter(adapter)
    head.output_shape = (1,)
    head.build(kerastuner.HyperParameters(), input_module.Input(shape=(32,)).build())
    assert head.loss.name == "binary_crossentropy"


def test_three_classes():
    y = np.array(["a", "a", "c", "b"])
    head = head_module.ClassificationHead(name="a")
    adapter = head.get_adapter()
    adapter.fit_transform(y)
    head.config_from_adapter(adapter)
    assert head.loss.name == "categorical_crossentropy"


def test_multi_label_loss():
    head = head_module.ClassificationHead(name="a", multi_label=True, num_classes=8)
    head.output_shape = (8,)
    input_node = tf.keras.Input(shape=(5,))
    output_node = head.build(kerastuner.HyperParameters(), input_node)
    model = tf.keras.Model(input_node, output_node)
    assert model.layers[-1].activation.__name__ == "sigmoid"
    assert head.loss.name == "binary_crossentropy"


def test_clf_head_build_with_zero_dropout_return_tensor():
    block = head_module.ClassificationHead(dropout=0)
    block.output_shape = (8,)

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(5,), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_segmentation():
    y = np.array(["a", "a", "c", "b"])
    head = head_module.SegmentationHead(name="a")
    adapter = head.get_adapter()
    adapter.fit_transform(y)
    head.config_from_adapter(adapter)
    input_shape = (64, 64, 21)
    hp = kerastuner.HyperParameters()
    head = blocks.deserialize(blocks.serialize(head))
    head.build(hp, ak.Input(shape=input_shape).build())
