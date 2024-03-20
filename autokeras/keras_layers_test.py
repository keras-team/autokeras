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

from autokeras import keras_layers as layer_module


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
