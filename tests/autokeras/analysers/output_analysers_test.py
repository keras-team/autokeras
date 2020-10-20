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
import pytest
import tensorflow as tf

from autokeras.analysers import output_analysers
from tests import utils


def test_clf_head_one_hot_shape_error():
    analyser = output_analysers.ClassificationAnalyser(name="a", num_classes=9)
    dataset = tf.data.Dataset.from_tensor_slices(
        utils.generate_one_hot_labels(dtype="np", num_classes=10)
    ).batch(32)

    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()

    assert "Expect the target data for a to have shape" in str(info.value)


def test_clf_head_more_dim_error():
    analyser = output_analysers.ClassificationAnalyser(name="a", num_classes=9)
    dataset = tf.data.Dataset.from_tensor_slices(
        np.random.rand(100, 32, 32, 3)
    ).batch(32)

    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()

    assert "Expect the target data for a to have shape" in str(info.value)


def test_wrong_num_classes_error():
    analyser = output_analysers.ClassificationAnalyser(name="a", num_classes=5)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 3)).batch(32)

    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()

    assert "Expect the target data for a to have shape" in str(info.value)


def test_one_class_error():
    analyser = output_analysers.ClassificationAnalyser(name="a")
    dataset = tf.data.Dataset.from_tensor_slices(np.array(["a", "a", "a"])).batch(32)

    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()

    assert "Expect the target data for a to have at least 2 classes" in str(
        info.value
    )


def test_infer_ten_classes():
    analyser = output_analysers.ClassificationAnalyser(name="a")
    dataset = utils.generate_one_hot_labels(dtype="dataset", num_classes=10)

    for data in dataset:
        analyser.update(data)
    analyser.finalize()

    assert analyser.num_classes == 10


def test_infer_single_column_two_classes():
    analyser = output_analysers.ClassificationAnalyser(name="a")
    dataset = tf.data.Dataset.from_tensor_slices(np.random.randint(0, 2, 10)).batch(
        32
    )

    for data in dataset:
        analyser.update(data)
    analyser.finalize()

    assert analyser.num_classes == 2


def test_specify_five_classes():
    analyser = output_analysers.ClassificationAnalyser(name="a", num_classes=5)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 5)).batch(32)

    for data in dataset:
        analyser.update(data)
    analyser.finalize()

    assert analyser.num_classes == 5


def test_specify_two_classes_fit_single_column():
    analyser = output_analysers.ClassificationAnalyser(name="a", num_classes=2)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 1)).batch(32)

    for data in dataset:
        analyser.update(data)
    analyser.finalize()

    assert analyser.num_classes == 2


def test_multi_label_two_classes_has_two_columns():
    analyser = output_analysers.ClassificationAnalyser(name="a", multi_label=True)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 2)).batch(32)

    for data in dataset:
        analyser.update(data)
    analyser.finalize()

    assert analyser.encoded


def test_reg_with_specified_output_dim_error():
    analyser = output_analysers.RegressionAnalyser(name="a", output_dim=3)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 2)).batch(32)

    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()

    assert "Expect the target data for a to have shape" in str(info.value)


def test_reg_with_specified_output_dim_and_single_column_doesnt_crash():
    analyser = output_analysers.RegressionAnalyser(name="a", output_dim=1)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10)).batch(32)

    for data in dataset:
        analyser.update(data)
    analyser.finalize()
