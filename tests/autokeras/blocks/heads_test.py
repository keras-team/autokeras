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
from autokeras import nodes as input_module
from autokeras import preprocessors
from autokeras.blocks import heads as head_module


def test_two_classes_infer_binary_crossentropy():
    dataset = np.array(["a", "a", "a", "b"])
    head = head_module.ClassificationHead(name="a", shape=(1,))
    adapter = head.get_adapter()
    dataset = adapter.adapt(dataset, batch_size=32)
    analyser = head.get_analyser()
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    head.config_from_analyser(analyser)
    head.build(
        kerastuner.HyperParameters(),
        input_module.Input(shape=(32,)).build_node(kerastuner.HyperParameters()),
    )
    assert head.loss.name == "binary_crossentropy"


def test_three_classes_infer_categorical_crossentropy():
    dataset = np.array(["a", "a", "c", "b"])
    head = head_module.ClassificationHead(name="a", shape=(1,))
    adapter = head.get_adapter()
    dataset = adapter.adapt(dataset, batch_size=32)
    analyser = head.get_analyser()
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    head.config_from_analyser(analyser)
    head.build(
        kerastuner.HyperParameters(),
        input_module.Input(shape=(32,)).build_node(kerastuner.HyperParameters()),
    )
    assert head.loss.name == "categorical_crossentropy"


def test_multi_label_loss():
    head = head_module.ClassificationHead(
        name="a", multi_label=True, num_classes=8, shape=(8,)
    )
    input_node = tf.keras.Input(shape=(5,))
    output_node = head.build(kerastuner.HyperParameters(), input_node)
    model = tf.keras.Model(input_node, output_node)
    assert model.layers[-1].activation.__name__ == "sigmoid"
    assert head.loss.name == "binary_crossentropy"


def test_clf_head_get_sigmoid_postprocessor():
    head = head_module.ClassificationHead(name="a", multi_label=True)
    head._encoded = True
    head._encoded_for_sigmoid = True
    assert isinstance(
        head.get_hyper_preprocessors()[0].preprocessor,
        preprocessors.SigmoidPostprocessor,
    )


def test_clf_head_with_2_clases_get_label_encoder():
    head = head_module.ClassificationHead(name="a", num_classes=2)
    head._encoded = False
    head._labels = ["a", "b"]
    assert isinstance(
        head.get_hyper_preprocessors()[-1].preprocessor, preprocessors.LabelEncoder
    )


def test_clf_head_build_with_zero_dropout_return_tensor():
    block = head_module.ClassificationHead(dropout=0, shape=(8,))

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(5,), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_segmentation():
    dataset = np.array(["a", "a", "c", "b"])
    head = head_module.SegmentationHead(name="a", shape=(1,))
    adapter = head.get_adapter()
    dataset = adapter.adapt(dataset, batch_size=32)
    analyser = head.get_analyser()
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    head.config_from_analyser(analyser)
    head.build(
        kerastuner.HyperParameters(),
        ak.Input(shape=(32,)).build_node(kerastuner.HyperParameters()),
    )
