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

import copy

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from autokeras.analysers import input_analysers
from tests import utils


def test_structured_data_input_less_col_name_error():
    with pytest.raises(ValueError) as info:
        analyser = input_analysers.StructuredDataAnalyser(
            column_names=list(range(8))
        )
        dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(20, 10)).batch(
            32
        )
        for x in dataset:
            analyser.update(x)

        analyser.finalize()

    assert "Expect column_names to have length" in str(info.value)


def test_structured_data_infer_col_types():
    analyser = input_analysers.StructuredDataAnalyser(
        column_names=utils.COLUMN_NAMES,
        column_types=None,
    )
    x = pd.read_csv(utils.TRAIN_CSV_PATH)
    x.pop("survived")
    dataset = tf.data.Dataset.from_tensor_slices(x.values.astype(np.unicode)).batch(
        32
    )

    for data in dataset:
        analyser.update(data)
    analyser.finalize()

    assert analyser.column_types == utils.COLUMN_TYPES


def test_dont_infer_specified_column_types():
    column_types = copy.copy(utils.COLUMN_TYPES)
    column_types.pop("sex")
    column_types["age"] = "categorical"

    analyser = input_analysers.StructuredDataAnalyser(
        column_names=utils.COLUMN_NAMES,
        column_types=column_types,
    )
    x = pd.read_csv(utils.TRAIN_CSV_PATH)
    x.pop("survived")
    dataset = tf.data.Dataset.from_tensor_slices(x.values.astype(np.unicode)).batch(
        32
    )

    for data in dataset:
        analyser.update(data)
    analyser.finalize()

    assert analyser.column_types["age"] == "categorical"


def test_structured_data_input_with_illegal_dim():
    analyser = input_analysers.StructuredDataAnalyser(
        column_names=utils.COLUMN_NAMES,
        column_types=None,
    )
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32, 32)).batch(
        32
    )

    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()

    assert "Expect the data to StructuredDataInput to have shape" in str(info.value)


def test_image_input_analyser_shape_is_list_of_int():
    analyser = input_analysers.ImageAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(
        np.random.rand(100, 32, 32, 3)
    ).batch(32)

    for data in dataset:
        analyser.update(data)
    analyser.finalize()

    assert isinstance(analyser.shape, list)
    assert all(map(lambda x: isinstance(x, int), analyser.shape))


def test_image_input_with_three_dim():
    analyser = input_analysers.ImageAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32, 32)).batch(
        32
    )

    for data in dataset:
        analyser.update(data)
    analyser.finalize()

    assert len(analyser.shape) == 3


def test_image_input_with_illegal_dim():
    analyser = input_analysers.ImageAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32)).batch(32)

    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()

    assert "Expect the data to ImageInput to have shape" in str(info.value)


def test_text_input_with_illegal_dim():
    analyser = input_analysers.TextAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32)).batch(32)

    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()

    assert "Expect the data to TextInput to have shape" in str(info.value)


def test_text_analyzer_with_one_dim_doesnt_crash():
    analyser = input_analysers.TextAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(["a b c", "b b c"]).batch(32)

    for data in dataset:
        analyser.update(data)
    analyser.finalize()


def test_text_illegal_type_error():
    analyser = input_analysers.TextAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 1)).batch(32)

    with pytest.raises(TypeError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()

    assert "Expect the data to TextInput to be strings" in str(info.value)


def test_time_series_input_with_illegal_dim():
    analyser = input_analysers.TimeseriesAnalyser(
        column_names=utils.COLUMN_NAMES,
        column_types=None,
    )
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32, 32)).batch(
        32
    )

    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()

    assert "Expect the data to TimeseriesInput to have shape" in str(info.value)
