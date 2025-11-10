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

from typing import Optional

import keras
import tree

from autokeras.blocks import basic
from autokeras.blocks import preprocessing
from autokeras.blocks import reduction
from autokeras.engine import block as block_module
from autokeras.utils import utils

BLOCK_TYPE = "block_type"
RESNET = "resnet"
XCEPTION = "xception"
VANILLA = "vanilla"
EFFICIENT = "efficient"
NORMALIZE = "normalize"
AUGMENT = "augment"
MAX_TOKENS = "max_tokens"


@keras.utils.register_keras_serializable(package="autokeras")
class ImageBlock(block_module.Block):
    """Block for image data.

    The image blocks is a block choosing from ResNetBlock, XceptionBlock,
    ConvBlock, which is controlled by a hyperparameter, 'block_type'.

    # Arguments
        block_type: String. 'resnet', 'xception', 'vanilla'. The type of Block
            to use. If unspecified, it will be tuned automatically.
        normalize: Boolean. Whether to channel-wise normalize the images.
            If unspecified, it will be tuned automatically.
        augment: Boolean. Whether to do image augmentation. If unspecified,
            it will be tuned automatically.
    """

    def __init__(
        self,
        block_type: Optional[str] = None,
        normalize: Optional[bool] = None,
        augment: Optional[bool] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_type = block_type
        self.normalize = normalize
        self.augment = augment

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                BLOCK_TYPE: self.block_type,
                NORMALIZE: self.normalize,
                AUGMENT: self.augment,
            }
        )
        return config

    def _build_block(self, hp, output_node, block_type):
        if block_type == RESNET:
            return basic.ResNetBlock().build(hp, output_node)
        elif block_type == XCEPTION:
            return basic.XceptionBlock().build(hp, output_node)
        elif block_type == VANILLA:
            return basic.ConvBlock().build(hp, output_node)
        elif block_type == EFFICIENT:
            return basic.EfficientNetBlock().build(hp, output_node)

    def build(self, hp, inputs=None):
        input_node = tree.flatten(inputs)[0]
        output_node = input_node

        if self.normalize is None and hp.Boolean(NORMALIZE):
            with hp.conditional_scope(NORMALIZE, [True]):
                output_node = preprocessing.Normalization().build(
                    hp, output_node
                )
        elif self.normalize:
            output_node = preprocessing.Normalization().build(hp, output_node)

        if self.augment is None and hp.Boolean(AUGMENT):
            with hp.conditional_scope(AUGMENT, [True]):
                output_node = preprocessing.ImageAugmentation().build(
                    hp, output_node
                )
        elif self.augment:
            output_node = preprocessing.ImageAugmentation().build(
                hp, output_node
            )

        if self.block_type is None:
            block_type = hp.Choice(
                BLOCK_TYPE, [RESNET, XCEPTION, VANILLA, EFFICIENT]
            )
            with hp.conditional_scope(BLOCK_TYPE, [block_type]):
                output_node = self._build_block(hp, output_node, block_type)
        else:
            output_node = self._build_block(hp, output_node, self.block_type)

        return output_node


@keras.utils.register_keras_serializable(package="autokeras")
class TextBlock(block_module.Block):
    """Block for text data.

    # Arguments
        max_tokens: Int. The maximum size of the vocabulary.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, max_tokens=None, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens

    def build(self, hp, inputs=None):
        input_node = tree.flatten(inputs)[0]
        output_node = input_node
        output_node = self._build_block(hp, output_node)
        return output_node

    def get_config(self):
        config = super().get_config()
        config.update({"max_tokens": self.max_tokens})
        return config

    def _build_block(self, hp, output_node):
        # Use Embedding and dense layers for tokenized text
        max_tokens = self.max_tokens or hp.Choice(
            MAX_TOKENS, [500, 5000, 20000], default=5000
        )
        output_node = basic.Embedding(
            max_features=max_tokens + 1,
        ).build(hp, output_node)
        output_node = basic.ConvBlock().build(hp, output_node)
        output_node = reduction.SpatialReduction().build(hp, output_node)
        output_node = basic.DenseBlock().build(hp, output_node)
        return output_node


@keras.utils.register_keras_serializable(package="autokeras")
class GeneralBlock(block_module.Block):
    """A general neural network block when the input type is unknown.

    When the input type is unknown. The GeneralBlock would search in a large
    space for a good model.

    # Arguments
        name: String.
    """

    def build(self, hp, inputs=None):
        inputs = tree.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        output_node = reduction.Flatten().build(hp, output_node)
        output_node = basic.DenseBlock().build(hp, output_node)
        return output_node


@keras.utils.register_keras_serializable(package="autokeras")
class StructuredDataBlock(block_module.Block):
    """Block for structured data.

    # Arguments
        categorical_encoding: Boolean. Whether to use the CategoricalToNumerical
            to encode the categorical features to numerical features. Defaults
            to True.
        normalize: Boolean. Whether to normalize the features.
            If unspecified, it will be tuned automatically.
    """

    def __init__(self, normalize: Optional[bool] = None, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "normalize": self.normalize,
            }
        )
        return config

    def build(self, hp, inputs=None):
        input_node = tree.flatten(inputs)[0]
        output_node = input_node

        if self.normalize is None and hp.Boolean(NORMALIZE):
            with hp.conditional_scope(NORMALIZE, [True]):
                output_node = preprocessing.Normalization().build(
                    hp, output_node
                )
        elif self.normalize:
            output_node = preprocessing.Normalization().build(hp, output_node)

        output_node = basic.DenseBlock().build(hp, output_node)
        return output_node
