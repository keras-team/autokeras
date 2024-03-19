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
from typing import Union

import keras_nlp
import tree
from keras import applications
from keras import layers
from keras import ops
from keras_tuner.engine import hyperparameters

from autokeras.blocks import reduction
from autokeras.engine import block as block_module
from autokeras.utils import io_utils
from autokeras.utils import layer_utils
from autokeras.utils import utils

RESNET_V1 = {
    "resnet50": applications.ResNet50,
    "resnet101": applications.ResNet101,
    "resnet152": applications.ResNet152,
}

RESNET_V2 = {
    "resnet50_v2": applications.ResNet50V2,
    "resnet101_v2": applications.ResNet101V2,
    "resnet152_v2": applications.ResNet152V2,
}

EFFICIENT_VERSIONS = {
    "b0": applications.EfficientNetB0,
    "b1": applications.EfficientNetB1,
    "b2": applications.EfficientNetB2,
    "b3": applications.EfficientNetB3,
    "b4": applications.EfficientNetB4,
    "b5": applications.EfficientNetB5,
    "b6": applications.EfficientNetB6,
    "b7": applications.EfficientNetB7,
}

PRETRAINED = "pretrained"


class DenseBlock(block_module.Block):
    """Block for Dense layers.

    # Arguments
        num_layers: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of Dense layers in the block.
            If left unspecified, it will be tuned automatically.
        num_units: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of units in each dense layer.
            If left unspecified, it will be tuned automatically.
        use_bn: Boolean. Whether to use BatchNormalization layers.
            If left unspecified, it will be tuned automatically.
        dropout: Float or keras_tuner.engine.hyperparameters.Choice.
            The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        num_layers: Optional[Union[int, hyperparameters.Choice]] = None,
        num_units: Optional[Union[int, hyperparameters.Choice]] = None,
        use_batchnorm: Optional[bool] = None,
        dropout: Optional[Union[float, hyperparameters.Choice]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = utils.get_hyperparameter(
            num_layers,
            hyperparameters.Choice("num_layers", [1, 2, 3], default=2),
            int,
        )
        self.num_units = utils.get_hyperparameter(
            num_units,
            hyperparameters.Choice(
                "num_units", [16, 32, 64, 128, 256, 512, 1024], default=32
            ),
            int,
        )
        self.use_batchnorm = use_batchnorm
        self.dropout = utils.get_hyperparameter(
            dropout,
            hyperparameters.Choice("dropout", [0.0, 0.25, 0.5], default=0.0),
            float,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": io_utils.serialize_block_arg(self.num_layers),
                "num_units": io_utils.serialize_block_arg(self.num_units),
                "use_batchnorm": self.use_batchnorm,
                "dropout": io_utils.serialize_block_arg(self.dropout),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["num_layers"] = io_utils.deserialize_block_arg(
            config["num_layers"]
        )
        config["num_units"] = io_utils.deserialize_block_arg(
            config["num_units"]
        )
        config["dropout"] = io_utils.deserialize_block_arg(config["dropout"])
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = tree.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        output_node = reduction.Flatten().build(hp, output_node)

        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Boolean("use_batchnorm", default=False)

        for i in range(utils.add_to_hp(self.num_layers, hp)):
            units = utils.add_to_hp(self.num_units, hp, "units_{i}".format(i=i))
            output_node = layers.Dense(units)(output_node)
            if use_batchnorm:
                output_node = layers.BatchNormalization()(output_node)
            output_node = layers.ReLU()(output_node)
            if utils.add_to_hp(self.dropout, hp) > 0:
                output_node = layers.Dropout(utils.add_to_hp(self.dropout, hp))(
                    output_node
                )
        return output_node


class RNNBlock(block_module.Block):
    """An RNN Block.

    # Arguments
        return_sequences: Boolean. Whether to return the last output in the
            output sequence, or the full sequence. Defaults to False.
        bidirectional: Boolean or keras_tuner.engine.hyperparameters.Boolean.
            Bidirectional RNN. If left unspecified, it will be
            tuned automatically.
        num_layers: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of layers in RNN. If left unspecified, it will
            be tuned automatically.
        layer_type: String or or keras_tuner.engine.hyperparameters.Choice.
            'gru' or 'lstm'. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(
        self,
        return_sequences: bool = False,
        bidirectional: Optional[Union[bool, hyperparameters.Boolean]] = None,
        num_layers: Optional[Union[int, hyperparameters.Choice]] = None,
        layer_type: Optional[Union[str, hyperparameters.Choice]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.return_sequences = return_sequences
        self.bidirectional = utils.get_hyperparameter(
            bidirectional,
            hyperparameters.Boolean("bidirectional", default=True),
            bool,
        )
        self.num_layers = utils.get_hyperparameter(
            num_layers,
            hyperparameters.Choice("num_layers", [1, 2, 3], default=2),
            int,
        )
        self.layer_type = utils.get_hyperparameter(
            layer_type,
            hyperparameters.Choice(
                "layer_type", ["gru", "lstm"], default="lstm"
            ),
            str,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "return_sequences": self.return_sequences,
                "bidirectional": io_utils.serialize_block_arg(
                    self.bidirectional
                ),
                "num_layers": io_utils.serialize_block_arg(self.num_layers),
                "layer_type": io_utils.serialize_block_arg(self.layer_type),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["bidirectional"] = io_utils.deserialize_block_arg(
            config["bidirectional"]
        )
        config["num_layers"] = io_utils.deserialize_block_arg(
            config["num_layers"]
        )
        config["layer_type"] = io_utils.deserialize_block_arg(
            config["layer_type"]
        )
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = tree.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        shape = list(input_node.shape)
        if len(shape) != 3:
            raise ValueError(
                "Expect the input tensor of RNNBlock to have dimensions of "
                "[batch_size, time_steps, vec_len], "
                "but got {shape}".format(shape=input_node.shape)
            )

        feature_size = shape[-1]
        output_node = input_node

        bidirectional = utils.add_to_hp(self.bidirectional, hp)
        layer_type = utils.add_to_hp(self.layer_type, hp)
        num_layers = utils.add_to_hp(self.num_layers, hp)
        rnn_layers = {"gru": layers.GRU, "lstm": layers.LSTM}
        in_layer = rnn_layers[layer_type]
        for i in range(num_layers):
            return_sequences = True
            if i == num_layers - 1:
                return_sequences = self.return_sequences
            if bidirectional:
                output_node = layers.Bidirectional(
                    in_layer(feature_size, return_sequences=return_sequences)
                )(output_node)
            else:
                output_node = in_layer(
                    feature_size, return_sequences=return_sequences
                )(output_node)
        return output_node


class ConvBlock(block_module.Block):
    """Block for vanilla ConvNets.

    # Arguments
        kernel_size: Int or keras_tuner.engine.hyperparameters.Choice.
            The size of the kernel.
            If left unspecified, it will be tuned automatically.
        num_blocks: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of conv blocks, each of which may contain
            convolutional, max pooling, dropout, and activation. If left
            unspecified, it will be tuned automatically.
        num_layers: Int or hyperparameters.Choice.
            The number of convolutional layers in each block. If left
            unspecified, it will be tuned automatically.
        filters: Int or keras_tuner.engine.hyperparameters.Choice. The number of
            filters in the convolutional layers. If left unspecified, it will
            be tuned automatically.
        max_pooling: Boolean. Whether to use max pooling layer in each block. If
            left unspecified, it will be tuned automatically.
        separable: Boolean. Whether to use separable conv layers.
            If left unspecified, it will be tuned automatically.
        dropout: Float or kerastuner.engine.hyperparameters.
            Choice range Between 0 and 1.
            The dropout rate after convolutional layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        kernel_size: Optional[Union[int, hyperparameters.Choice]] = None,
        num_blocks: Optional[Union[int, hyperparameters.Choice]] = None,
        num_layers: Optional[Union[int, hyperparameters.Choice]] = None,
        filters: Optional[Union[int, hyperparameters.Choice]] = None,
        max_pooling: Optional[bool] = None,
        separable: Optional[bool] = None,
        dropout: Optional[Union[float, hyperparameters.Choice]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_size = utils.get_hyperparameter(
            kernel_size,
            hyperparameters.Choice("kernel_size", [3, 5, 7], default=3),
            int,
        )
        self.num_blocks = utils.get_hyperparameter(
            num_blocks,
            hyperparameters.Choice("num_blocks", [1, 2, 3], default=2),
            int,
        )
        self.num_layers = utils.get_hyperparameter(
            num_layers,
            hyperparameters.Choice("num_layers", [1, 2], default=2),
            int,
        )
        self.filters = utils.get_hyperparameter(
            filters,
            hyperparameters.Choice(
                "filters", [16, 32, 64, 128, 256, 512], default=32
            ),
            int,
        )
        self.max_pooling = max_pooling
        self.separable = separable
        self.dropout = utils.get_hyperparameter(
            dropout,
            hyperparameters.Choice("dropout", [0.0, 0.25, 0.5], default=0.0),
            float,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": io_utils.serialize_block_arg(self.kernel_size),
                "num_blocks": io_utils.serialize_block_arg(self.num_blocks),
                "num_layers": io_utils.serialize_block_arg(self.num_layers),
                "filters": io_utils.serialize_block_arg(self.filters),
                "max_pooling": self.max_pooling,
                "separable": self.separable,
                "dropout": io_utils.serialize_block_arg(self.dropout),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["kernel_size"] = io_utils.deserialize_block_arg(
            config["kernel_size"]
        )
        config["num_blocks"] = io_utils.deserialize_block_arg(
            config["num_blocks"]
        )
        config["num_layers"] = io_utils.deserialize_block_arg(
            config["num_layers"]
        )
        config["filters"] = io_utils.deserialize_block_arg(config["filters"])
        config["dropout"] = io_utils.deserialize_block_arg(config["dropout"])
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = tree.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        kernel_size = utils.add_to_hp(self.kernel_size, hp)

        separable = self.separable
        if separable is None:
            separable = hp.Boolean("separable", default=False)

        if separable:
            conv = layer_utils.get_sep_conv(input_node.shape)
        else:
            conv = layer_utils.get_conv(input_node.shape)

        max_pooling = self.max_pooling
        if max_pooling is None:
            max_pooling = hp.Boolean("max_pooling", default=True)
        pool = layer_utils.get_max_pooling(input_node.shape)

        for i in range(utils.add_to_hp(self.num_blocks, hp)):
            for j in range(utils.add_to_hp(self.num_layers, hp)):
                output_node = conv(
                    utils.add_to_hp(
                        self.filters, hp, "filters_{i}_{j}".format(i=i, j=j)
                    ),
                    kernel_size,
                    padding=self._get_padding(kernel_size, output_node),
                    activation="relu",
                )(output_node)
            if max_pooling:
                output_node = pool(
                    kernel_size - 1,
                    padding=self._get_padding(kernel_size - 1, output_node),
                )(output_node)
            if utils.add_to_hp(self.dropout, hp) > 0:
                output_node = layers.Dropout(utils.add_to_hp(self.dropout, hp))(
                    output_node
                )
        return output_node

    @staticmethod
    def _get_padding(kernel_size, output_node):
        if all(kernel_size * 2 <= length for length in output_node.shape[1:-1]):
            return "valid"
        return "same"


class KerasApplicationBlock(block_module.Block):
    """Blocks extending Keras applications."""

    def __init__(self, pretrained, models, min_size, **kwargs):
        super().__init__(**kwargs)
        self.pretrained = pretrained
        self.models = models
        self.min_size = min_size

    def get_config(self):
        config = super().get_config()
        config.update({"pretrained": self.pretrained})
        return config

    def build(self, hp, inputs=None):
        input_node = tree.flatten(inputs)[0]

        pretrained = self.pretrained
        if input_node.shape[3] not in [1, 3]:
            if self.pretrained:
                raise ValueError(
                    "When pretrained is set to True, expect input to "
                    "have 1 or 3 channels, bug got "
                    "{channels}.".format(channels=input_node.shape[3])
                )
            pretrained = False
        if pretrained is None:
            pretrained = hp.Boolean(PRETRAINED, default=False)
            if pretrained:
                with hp.conditional_scope(PRETRAINED, [True]):
                    trainable = hp.Boolean("trainable", default=False)
        elif pretrained:
            trainable = hp.Boolean("trainable", default=False)

        if len(self.models) > 1:
            version = hp.Choice("version", list(self.models.keys()))
        else:
            version = list(self.models.keys())[0]

        min_size = self.min_size
        if hp.Boolean("imagenet_size", default=False):
            min_size = 224
        if input_node.shape[1] < min_size or input_node.shape[2] < min_size:
            input_node = layers.Resizing(
                max(min_size, input_node.shape[1]),
                max(min_size, input_node.shape[2]),
            )(input_node)
        if input_node.shape[3] == 1:
            input_node = layers.Concatenate()([input_node] * 3)
        if input_node.shape[3] != 3:
            input_node = layers.Conv2D(
                filters=3, kernel_size=1, padding="same"
            )(input_node)

        if pretrained:
            model = self.models[version](weights="imagenet", include_top=False)
            model.trainable = trainable
        else:
            model = self.models[version](
                weights=None,
                include_top=False,
                input_shape=input_node.shape[1:],
            )

        return model(input_node)


class ResNetBlock(KerasApplicationBlock):
    """Block for ResNet.

    # Arguments
        version: String. 'v1', 'v2'. The type of ResNet to use.
            If left unspecified, it will be tuned automatically.
        pretrained: Boolean. Whether to use ImageNet pretrained weights.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        version: Optional[str] = None,
        pretrained: Optional[bool] = None,
        **kwargs,
    ):
        if version is None:
            models = {**RESNET_V1, **RESNET_V2}
        elif version == "v1":
            models = RESNET_V1
        elif version == "v2":
            models = RESNET_V2
        else:
            raise ValueError(
                'Expect version to be "v1", or "v2", but got '
                "{version}.".format(version=version)
            )
        super().__init__(
            pretrained=pretrained, models=models, min_size=32, **kwargs
        )
        self.version = version

    def get_config(self):
        config = super().get_config()
        config.update({"version": self.version})
        return config


class XceptionBlock(KerasApplicationBlock):
    """Block for XceptionNet.

    An Xception structure, used for specifying your model with specific
    datasets.

    The original Xception architecture is from
    [https://arxiv.org/abs/1610.02357](https://arxiv.org/abs/1610.02357).
    The data first goes through the entry flow, then through the middle flow
    which is repeated eight times, and finally through the exit flow.

    This XceptionBlock returns a similar architecture as Xception except without
    the last (optional) fully connected layer(s) and logistic regression.
    The size of this architecture could be decided by `HyperParameters`, to get
    an architecture with a half, an identical, or a double size of the original
    one.

    # Arguments
        pretrained: Boolean. Whether to use ImageNet pretrained weights.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, pretrained: Optional[bool] = None, **kwargs):
        super().__init__(
            pretrained=pretrained,
            models={"xception": applications.Xception},
            min_size=71,
            **kwargs,
        )


class EfficientNetBlock(KerasApplicationBlock):
    """Block for EfficientNet.

    # Arguments
        version: String. The value should be one of 'b0', 'b1', ..., 'b7'. The
            type of EfficientNet to use. If left unspecified, it will be tuned
            automatically.
        pretrained: Boolean. Whether to use ImageNet pretrained weights.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        version: Optional[str] = None,
        pretrained: Optional[bool] = None,
        **kwargs,
    ):
        if version is None:
            models = EFFICIENT_VERSIONS
        elif version in EFFICIENT_VERSIONS.keys():
            models = {version: EFFICIENT_VERSIONS[version]}
        else:
            raise ValueError(
                "Expect version to be in {expect}, but got "
                "{version}.".format(
                    expect=list(EFFICIENT_VERSIONS.keys()), version=version
                )
            )
        super().__init__(
            pretrained=pretrained,
            models=models,
            min_size=32,
            **kwargs,
        )
        self.version = version


class BertBlock(block_module.Block):
    """Block for Pre-trained BERT.

    The input should be sequence of sentences without the padded tokens, like
    [CLS] [SEP] [PAD].

    # Example
    ```python
        # Using the Transformer Block with AutoModel.
        import autokeras as ak
        from autokeras import BertBlock
        from keras import losses

        input_node = ak.TextInput()
        output_node = BertBlock()(input_node)
        output_node = ak.ClassificationHead()(output_node)
        clf = ak.AutoModel(
            inputs=input_node, outputs=output_node, max_trials=10)
    ```

    # Arguments
        max_sequence_length: Int or keras_tuner.engine.hyperparameters.Choice.
            The maximum length of a sequence that is used to train the model.
    """

    def __init__(
        self,
        max_sequence_length: Optional[
            Union[int, hyperparameters.Choice]
        ] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_sequence_length = utils.get_hyperparameter(
            max_sequence_length,
            hyperparameters.Choice(
                "max_sequence_length", [128, 256, 512], default=128
            ),
            int,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_sequence_length": io_utils.serialize_block_arg(
                    self.max_sequence_length
                )
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["max_sequence_length"] = io_utils.deserialize_block_arg(
            config["max_sequence_length"]
        )
        return cls(**config)

    def build(self, hp, inputs=None):
        input_tensor = tree.flatten(inputs)[0]

        preset_name = "bert_base_en_uncased"
        tokenizer_layer = keras_nlp.models.BertPreprocessor.from_preset(
            preset_name,
            sequence_length=utils.add_to_hp(self.max_sequence_length, hp),
        )
        bert_encoder = keras_nlp.models.BertBackbone.from_preset(preset_name)

        output_node = tokenizer_layer(ops.reshape(input_tensor, [-1]))
        output_node = bert_encoder(output_node)["pooled_output"]

        return output_node
