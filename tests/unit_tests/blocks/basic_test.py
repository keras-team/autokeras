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
import pytest
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import blocks
from tests import utils


def test_resnet_build_return_tensor():
    block = blocks.ResNetBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_resnet_v1_return_tensor():
    block = blocks.ResNetBlock(version="v1")

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_efficientnet_b0_return_tensor():
    block = blocks.EfficientNetBlock(version="b0", pretrained=False)

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_resnet_pretrained_build_return_tensor():
    block = blocks.ResNetBlock(pretrained=True)

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_resnet_pretrained_with_one_channel_input():
    block = blocks.ResNetBlock(pretrained=True)

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(28, 28, 1), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_resnet_pretrained_error_with_two_channels():
    block = blocks.ResNetBlock(pretrained=True)

    with pytest.raises(ValueError) as info:
        block.build(
            kerastuner.HyperParameters(),
            tf.keras.Input(shape=(224, 224, 2), dtype=tf.float32),
        )

    assert "When pretrained is set to True" in str(info.value)


def test_resnet_deserialize_to_resnet():
    serialized_block = blocks.serialize(blocks.ResNetBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.ResNetBlock)


def test_resnet_get_config_has_all_attributes():
    block = blocks.ResNetBlock()

    config = block.get_config()

    assert utils.get_func_args(blocks.ResNetBlock.__init__).issubset(config.keys())


def test_resnet_wrong_version_error():
    with pytest.raises(ValueError) as info:
        blocks.ResNetBlock(version="abc")

    assert "Expect version to be" in str(info.value)


def test_efficientnet_wrong_version_error():
    with pytest.raises(ValueError) as info:
        blocks.EfficientNetBlock(version="abc")

    assert "Expect version to be" in str(info.value)


def test_xception_build_return_tensor():
    block = blocks.XceptionBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 2), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_xception_pretrained_build_return_tensor():
    block = blocks.XceptionBlock(pretrained=True)

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_xception_pretrained_with_one_channel_input():
    block = blocks.XceptionBlock(pretrained=True)

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(224, 224, 1), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_xception_pretrained_error_with_two_channels():
    block = blocks.XceptionBlock(pretrained=True)

    with pytest.raises(ValueError) as info:
        block.build(
            kerastuner.HyperParameters(),
            tf.keras.Input(shape=(224, 224, 2), dtype=tf.float32),
        )

    assert "When pretrained is set to True" in str(info.value)


def test_xception_deserialize_to_xception():
    serialized_block = blocks.serialize(blocks.XceptionBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.XceptionBlock)


def test_xception_get_config_has_all_attributes():
    block = blocks.XceptionBlock()

    config = block.get_config()

    assert utils.get_func_args(blocks.XceptionBlock.__init__).issubset(config.keys())


def test_conv_build_return_tensor():
    block = blocks.ConvBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_conv_with_small_image_size_return_tensor():
    block = blocks.ConvBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(10, 10, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_conv_build_with_dropout_return_tensor():
    block = blocks.ConvBlock(dropout=0.5)

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_conv_deserialize_to_conv():
    serialized_block = blocks.serialize(blocks.ConvBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.ConvBlock)


def test_conv_get_config_has_all_attributes():
    block = blocks.ConvBlock()

    config = block.get_config()

    assert utils.get_func_args(blocks.ConvBlock.__init__).issubset(config.keys())


def test_rnn_build_return_tensor():
    block = blocks.RNNBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 10), dtype=tf.float32),
    )

    assert len(nest.flatten(outputs)) == 1


def test_rnn_input_shape_one_dim_error():
    block = blocks.RNNBlock()

    with pytest.raises(ValueError) as info:
        block.build(
            kerastuner.HyperParameters(),
            tf.keras.Input(shape=(32,), dtype=tf.float32),
        )

    assert "Expect the input tensor of RNNBlock" in str(info.value)


def test_rnn_deserialize_to_rnn():
    serialized_block = blocks.serialize(blocks.RNNBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.RNNBlock)


def test_rnn_get_config_has_all_attributes():
    block = blocks.RNNBlock()

    config = block.get_config()

    assert utils.get_func_args(blocks.RNNBlock.__init__).issubset(config.keys())


def test_dense_build_return_tensor():
    block = blocks.DenseBlock(
        num_units=kerastuner.engine.hyperparameters.Choice("num_units", [10, 20])
    )

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(32,), dtype=tf.float32)
    )

    assert len(nest.flatten(outputs)) == 1


def test_dense_build_with_dropout_return_tensor():
    block = blocks.DenseBlock(dropout=0.5)

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(32,), dtype=tf.float32)
    )

    assert len(nest.flatten(outputs)) == 1


def test_dense_build_with_bn_return_tensor():
    block = blocks.DenseBlock(use_batchnorm=True)

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(32,), dtype=tf.float32)
    )

    assert len(nest.flatten(outputs)) == 1


def test_dense_deserialize_to_dense():
    serialized_block = blocks.serialize(blocks.DenseBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.DenseBlock)


def test_dense_get_config_has_all_attributes():
    block = blocks.DenseBlock()

    config = block.get_config()

    assert utils.get_func_args(blocks.DenseBlock.__init__).issubset(config.keys())


def test_embed_build_return_tensor():
    block = blocks.Embedding()

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(32,), dtype=tf.float32)
    )

    assert len(nest.flatten(outputs)) == 1


def test_embed_deserialize_to_embed():
    serialized_block = blocks.serialize(blocks.Embedding())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.Embedding)


def test_embed_get_config_has_all_attributes():
    block = blocks.Embedding()

    config = block.get_config()

    assert utils.get_func_args(blocks.Embedding.__init__).issubset(config.keys())


def test_transformer_build_return_tensor():
    block = blocks.Transformer()

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(64,), dtype=tf.float32)
    )

    assert len(nest.flatten(outputs)) == 1


def test_transformer_deserialize_to_transformer():
    serialized_block = blocks.serialize(blocks.Transformer())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.Transformer)


def test_transformer_get_config_has_all_attributes():
    block = blocks.Transformer()

    config = block.get_config()

    assert utils.get_func_args(blocks.Transformer.__init__).issubset(config.keys())


def test_multi_head_restore_head_size():
    block = blocks.basic.MultiHeadSelfAttention(head_size=16)

    block = blocks.basic.MultiHeadSelfAttention.from_config(block.get_config())

    assert block.head_size == 16


def test_bert_build_return_tensor():
    block = blocks.BertBlock()

    outputs = block.build(
        kerastuner.HyperParameters(), tf.keras.Input(shape=(1,), dtype=tf.string)
    )

    assert len(nest.flatten(outputs)) == 1


def test_bert_deserialize_to_transformer():
    serialized_block = blocks.serialize(blocks.BertBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.BertBlock)


def test_bert_get_config_has_all_attributes():
    block = blocks.BertBlock()

    config = block.get_config()

    assert utils.get_func_args(blocks.BertBlock.__init__).issubset(config.keys())
