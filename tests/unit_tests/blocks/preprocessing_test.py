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


def test_augment_build_return_tensor():
    block = blocks.ImageAugmentation()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_augment_build_with_translation_factor_range_return_tensor():
    block = blocks.ImageAugmentation(translation_factor=(0, 0.1))

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_augment_build_with_no_flip_return_tensor():
    block = blocks.ImageAugmentation(vertical_flip=False, horizontal_flip=False)

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_augment_build_with_vflip_only_return_tensor():
    block = blocks.ImageAugmentation(vertical_flip=True, horizontal_flip=False)

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_augment_build_with_zoom_factor_return_tensor():
    block = blocks.ImageAugmentation(zoom_factor=0.1)

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_augment_build_with_contrast_factor_return_tensor():
    block = blocks.ImageAugmentation(contrast_factor=0.1)

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_augment_deserialize_to_augment():
    serialized_block = blocks.serialize(blocks.ImageAugmentation())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.ImageAugmentation)


def test_augment_get_config_has_all_attributes():
    block = blocks.ImageAugmentation()

    config = block.get_config()

    assert utils.get_func_args(blocks.ImageAugmentation.__init__).issubset(
        config.keys()
    )


def test_ngram_build_return_tensor():
    block = blocks.TextToNgramVector()

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(1,), dtype=tf.string)
    )

    assert len(nest.flatten(outputs)) == 1


def test_ngram_build_with_ngrams_return_tensor():
    block = blocks.TextToNgramVector(ngrams=2)

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(1,), dtype=tf.string)
    )

    assert len(nest.flatten(outputs)) == 1


def test_ngram_deserialize_to_ngram():
    serialized_block = blocks.serialize(blocks.TextToNgramVector())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.TextToNgramVector)


def test_ngram_get_config_has_all_attributes():
    block = blocks.TextToNgramVector()

    config = block.get_config()

    assert utils.get_func_args(blocks.TextToNgramVector.__init__).issubset(
        config.keys()
    )


def test_int_seq_build_return_tensor():
    block = blocks.TextToIntSequence()

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(1,), dtype=tf.string)
    )

    assert len(nest.flatten(outputs)) == 1


def test_int_seq_build_with_seq_len_return_tensor():
    block = blocks.TextToIntSequence(output_sequence_length=50)

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(1,), dtype=tf.string)
    )

    assert len(nest.flatten(outputs)) == 1


def test_int_seq_deserialize_to_int_seq():
    serialized_block = blocks.serialize(blocks.TextToIntSequence())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.TextToIntSequence)


def test_int_seq_get_config_has_all_attributes():
    block = blocks.TextToIntSequence()

    config = block.get_config()

    assert utils.get_func_args(blocks.TextToIntSequence.__init__).issubset(
        config.keys()
    )


def test_cat_to_num_build_return_tensor():
    block = blocks.CategoricalToNumerical()
    block.column_names = ["a"]
    block.column_types = {"a": "num"}

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(1,), dtype=tf.string)
    )

    assert len(nest.flatten(outputs)) == 1


def test_cat_to_num_deserialize_to_cat_to_num():
    serialized_block = blocks.serialize(blocks.CategoricalToNumerical())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.CategoricalToNumerical)


def test_cat_to_num_get_config_has_all_attributes():
    block = blocks.CategoricalToNumerical()

    config = block.get_config()

    assert utils.get_func_args(blocks.CategoricalToNumerical.__init__).issubset(
        config.keys()
    )
