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

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from autokeras import keras_layers as layer_module


def test_multi_cat_encode_strings_correctly(tmp_path):
    x_train = np.array([["a", "ab", 2.1], ["b", "bc", 1.0], ["a", "bc", "nan"]])
    layer = layer_module.MultiCategoryEncoding(
        [layer_module.INT, layer_module.INT, layer_module.NONE]
    )
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(32)

    layer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(32))

    for data in dataset:
        result = layer(data)

    assert result[0][0] == result[2][0]
    assert result[0][0] != result[1][0]
    assert result[0][1] != result[1][1]
    assert result[0][1] != result[2][1]
    assert result[2][2] == 0
    assert result.dtype == tf.float32


def test_model_save_load_output_same(tmp_path):
    x_train = np.array([["a", "ab", 2.1], ["b", "bc", 1.0], ["a", "bc", "nan"]])
    layer = layer_module.MultiCategoryEncoding(
        encoding=[layer_module.INT, layer_module.INT, layer_module.NONE]
    )
    layer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(32))

    model = keras.Sequential([keras.Input(shape=(3,), dtype=tf.string), layer])
    model.save(os.path.join(tmp_path, "model"))
    model2 = keras.models.load_model(os.path.join(tmp_path, "model"))

    assert np.array_equal(model.predict(x_train), model2.predict(x_train))


def test_init_multi_one_hot_encode():
    layer_module.MultiCategoryEncoding(
        encoding=[layer_module.ONE_HOT, layer_module.INT, layer_module.NONE]
    )
    # TODO: add more content when it is implemented


def test_call_multi_with_single_column_return_right_shape():
    x_train = np.array([["a"], ["b"], ["a"]])
    layer = layer_module.MultiCategoryEncoding(encoding=[layer_module.INT])
    layer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(32))

    assert layer(x_train).shape == (3, 1)


def get_text_data():
    train = np.array(
        [
            ["This is a test example"],
            ["This is another text example"],
            ["Is this another example?"],
            [""],
            ["Is this a long long long long long long example?"],
        ],
        dtype=str,
    )
    test = np.array(
        [
            ["This is a test example"],
            ["This is another text example"],
            ["Is this another example?"],
        ],
        dtype=str,
    )
    y = np.random.rand(3, 1)
    return train, test, y


def test_cast_to_float32_return_float32_tensor(tmp_path):
    layer = layer_module.CastToFloat32()

    tensor = layer(tf.constant(["0.3"], dtype=tf.string))

    assert tf.float32 == tensor.dtype


def test_expand_last_dim_return_tensor_with_more_dims(tmp_path):
    layer = layer_module.ExpandLastDim()

    tensor = layer(tf.constant([0.1, 0.2], dtype=tf.float32))

    assert 2 == len(tensor.shape.as_list())
