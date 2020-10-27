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

from autokeras.preprocessors import common
from autokeras.utils import data_utils


def test_time_series_input_transform():
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32)).batch(32)
    preprocessor = common.SlidingWindow(lookback=2, batch_size=32)
    x = preprocessor.transform(dataset)
    assert data_utils.dataset_shape(x).as_list() == [None, 2, 32]


def test_categorical_to_numerical_input_transform():
    x_train = np.array([["a", "ab", 2.1], ["b", "bc", 1.0], ["a", "bc", "nan"]])
    preprocessor = common.CategoricalToNumericalPreprocessor(
        column_names=["column_a", "column_b", "column_c"],
        column_types={
            "column_a": "categorical",
            "column_b": "categorical",
            "column_c": "numerical",
        },
    )
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(32)

    preprocessor.fit(tf.data.Dataset.from_tensor_slices(x_train).batch(32))
    result = preprocessor.transform(dataset)

    assert result[0][0] == result[2][0]
    assert result[0][0] != result[1][0]
    assert result[0][1] != result[1][1]
    assert result[0][1] != result[2][1]
    assert result[2][2] == 0
    assert result.dtype == tf.float32
