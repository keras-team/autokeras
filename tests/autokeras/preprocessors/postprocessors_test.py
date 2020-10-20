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

import numpy as np
import tensorflow as tf

from autokeras import preprocessors
from autokeras.preprocessors import postprocessors


def test_sigmoid_postprocess_to_zero_one():
    postprocessor = postprocessors.SigmoidPostprocessor()

    y = postprocessor.postprocess(np.random.rand(10, 3))

    assert set(y.flatten().tolist()) == set([1, 0])


def test_sigmoid_transform_dataset_doesnt_change():
    postprocessor = postprocessors.SigmoidPostprocessor()
    dataset = tf.data.Dataset.from_tensor_slices([1, 2]).batch(32)

    assert postprocessor.transform(dataset) is dataset


def test_sigmoid_deserialize_without_error():
    postprocessor = postprocessors.SigmoidPostprocessor()
    dataset = tf.data.Dataset.from_tensor_slices([1, 2]).batch(32)

    postprocessor = preprocessors.deserialize(preprocessors.serialize(postprocessor))

    assert postprocessor.transform(dataset) is dataset


def test_softmax_postprocess_to_zero_one():
    postprocessor = postprocessors.SoftmaxPostprocessor()

    y = postprocessor.postprocess(np.random.rand(10, 3))

    assert set(y.flatten().tolist()) == set([1, 0])


def test_softmax_transform_dataset_doesnt_change():
    postprocessor = postprocessors.SoftmaxPostprocessor()
    dataset = tf.data.Dataset.from_tensor_slices([1, 2]).batch(32)

    assert postprocessor.transform(dataset) is dataset


def test_softmax_deserialize_without_error():
    postprocessor = postprocessors.SoftmaxPostprocessor()
    dataset = tf.data.Dataset.from_tensor_slices([1, 2]).batch(32)

    postprocessor = preprocessors.deserialize(preprocessors.serialize(postprocessor))

    assert postprocessor.transform(dataset) is dataset
