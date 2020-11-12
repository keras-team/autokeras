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
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import blocks
from tests import utils


def test_merge_build_return_tensor():
    block = blocks.Merge()

    outputs = block.build(
        kerastuner.HyperParameters(),
        [
            tf.keras.Input(shape=(32,), dtype=tf.float32),
            tf.keras.Input(shape=(4, 8), dtype=tf.float32),
        ],
    )

    assert len(nest.flatten(outputs)) == 1


def test_merge_single_input_return_tensor():
    block = blocks.Merge()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32,), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_merge_inputs_with_same_shape_return_tensor():
    block = blocks.Merge()

    outputs = block.build(
        kerastuner.HyperParameters(),
        [
            tf.keras.Input(shape=(32,), dtype=tf.float32),
            tf.keras.Input(shape=(32,), dtype=tf.float32),
        ],
    )

    assert len(nest.flatten(outputs)) == 1


def test_merge_deserialize_to_merge():
    serialized_block = blocks.serialize(blocks.Merge())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.Merge)


def test_merge_get_config_has_all_attributes():
    block = blocks.Merge()

    config = block.get_config()

    assert utils.get_func_args(blocks.Merge.__init__).issubset(config.keys())


def test_temporal_build_return_tensor():
    block = blocks.TemporalReduction()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 10), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_temporal_global_max_return_tensor():
    block = blocks.TemporalReduction(reduction_type="global_max")

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 10), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_temporal_global_avg_return_tensor():
    block = blocks.TemporalReduction(reduction_type="global_avg")

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 10), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_reduction_2d_tensor_return_input_node():
    block = blocks.TemporalReduction()
    input_node = tf.keras.Input(shape=(32,), dtype=tf.float32)

    outputs = block.build(
        kerastuner.HyperParameters(),
        input_node,
    )

    assert len(nest.flatten(outputs)) == 1
    assert nest.flatten(outputs)[0] is input_node


def test_temporal_deserialize_to_temporal():
    serialized_block = blocks.serialize(blocks.TemporalReduction())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.TemporalReduction)


def test_temporal_get_config_has_all_attributes():
    block = blocks.TemporalReduction()

    config = block.get_config()

    assert utils.get_func_args(blocks.TemporalReduction.__init__).issubset(
        config.keys()
    )


def test_spatial_build_return_tensor():
    block = blocks.SpatialReduction()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_spatial_deserialize_to_spatial():
    serialized_block = blocks.serialize(blocks.SpatialReduction())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.SpatialReduction)


def test_spatial_get_config_has_all_attributes():
    block = blocks.SpatialReduction()

    config = block.get_config()

    assert utils.get_func_args(blocks.SpatialReduction.__init__).issubset(
        config.keys()
    )
