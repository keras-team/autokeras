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

import keras
import keras_tuner
import tree
from keras_tuner.engine import hyperparameters

from autokeras import blocks
from autokeras import test_utils


def test_augment_build_return_tensor():
    block = blocks.ImageAugmentation(rotation_factor=0.2)

    outputs = block.build(
        keras_tuner.HyperParameters(),
        keras.Input(shape=(32, 32, 3), dtype="float32"),
    )

    assert len(tree.flatten(outputs)) == 1


def test_augment_build_with_translation_factor_range_return_tensor():
    block = blocks.ImageAugmentation(translation_factor=(0, 0.1))

    outputs = block.build(
        keras_tuner.HyperParameters(),
        keras.Input(shape=(32, 32, 3), dtype="float32"),
    )

    assert len(tree.flatten(outputs)) == 1


def test_augment_build_with_no_flip_return_tensor():
    block = blocks.ImageAugmentation(vertical_flip=False, horizontal_flip=False)

    outputs = block.build(
        keras_tuner.HyperParameters(),
        keras.Input(shape=(32, 32, 3), dtype="float32"),
    )

    assert len(tree.flatten(outputs)) == 1


def test_augment_build_with_vflip_only_return_tensor():
    block = blocks.ImageAugmentation(vertical_flip=True, horizontal_flip=False)

    outputs = block.build(
        keras_tuner.HyperParameters(),
        keras.Input(shape=(32, 32, 3), dtype="float32"),
    )

    assert len(tree.flatten(outputs)) == 1


def test_augment_build_with_zoom_factor_return_tensor():
    block = blocks.ImageAugmentation(zoom_factor=0.1)

    outputs = block.build(
        keras_tuner.HyperParameters(),
        keras.Input(shape=(32, 32, 3), dtype="float32"),
    )

    assert len(tree.flatten(outputs)) == 1


def test_augment_build_with_contrast_factor_return_tensor():
    block = blocks.ImageAugmentation(contrast_factor=0.1)

    outputs = block.build(
        keras_tuner.HyperParameters(),
        keras.Input(shape=(32, 32, 3), dtype="float32"),
    )

    assert len(tree.flatten(outputs)) == 1


def test_augment_deserialize_to_augment():
    serialized_block = blocks.serialize(
        blocks.ImageAugmentation(
            zoom_factor=0.1,
            contrast_factor=hyperparameters.Float("contrast_factor", 0.1, 0.5),
        )
    )

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.ImageAugmentation)
    assert block.zoom_factor == 0.1
    assert isinstance(block.contrast_factor, hyperparameters.Float)


def test_augment_get_config_has_all_attributes():
    block = blocks.ImageAugmentation()

    config = block.get_config()

    assert test_utils.get_func_args(blocks.ImageAugmentation.__init__).issubset(
        config.keys()
    )
