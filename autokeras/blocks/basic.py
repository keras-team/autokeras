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

import tensorflow as tf
from kerastuner.engine import hyperparameters
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.python.util import nest

from autokeras import keras_layers
from autokeras.blocks import reduction
from autokeras.engine import block as block_module
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
        num_layers: Int or kerastuner.engine.hyperparameters.Choice.
            The number of Dense layers in the block.
            If left unspecified, it will be tuned automatically.
        num_units: Int or kerastuner.engine.hyperparameters.Choice.
            The number of units in each dense layer.
            If left unspecified, it will be tuned automatically.
        use_bn: Boolean. Whether to use BatchNormalization layers.
            If left unspecified, it will be tuned automatically.
        dropout: Float or kerastuner.engine.hyperparameters.Choice.
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
                "num_layers": hyperparameters.serialize(self.num_layers),
                "num_units": hyperparameters.serialize(self.num_units),
                "use_batchnorm": self.use_batchnorm,
                "dropout": hyperparameters.serialize(self.dropout),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["num_layers"] = hyperparameters.deserialize(config["num_layers"])
        config["num_units"] = hyperparameters.deserialize(config["num_units"])
        config["dropout"] = hyperparameters.deserialize(config["dropout"])
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
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
        bidirectional: Boolean. Bidirectional RNN. If left unspecified, it will be
            tuned automatically.
        num_layers: Int. The number of layers in RNN. If left unspecified, it will
            be tuned automatically.
        layer_type: String. 'gru' or 'lstm'. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(
        self,
        return_sequences: bool = False,
        bidirectional: Optional[bool] = None,
        num_layers: Optional[int] = None,
        layer_type: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.layer_type = layer_type

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "return_sequences": self.return_sequences,
                "bidirectional": self.bidirectional,
                "num_layers": self.num_layers,
                "layer_type": self.layer_type,
            }
        )
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        shape = input_node.shape.as_list()
        if len(shape) != 3:
            raise ValueError(
                "Expect the input tensor of RNNBlock to have dimensions of "
                "[batch_size, time_steps, vec_len], "
                "but got {shape}".format(shape=input_node.shape)
            )

        feature_size = shape[-1]
        output_node = input_node

        bidirectional = self.bidirectional
        if bidirectional is None:
            bidirectional = hp.Boolean("bidirectional", default=True)
        layer_type = self.layer_type or hp.Choice(
            "layer_type", ["gru", "lstm"], default="lstm"
        )
        num_layers = self.num_layers or hp.Choice("num_layers", [1, 2, 3], default=2)
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
        kernel_size: Int or kerastuner.engine.hyperparameters.Choice.
            The size of the kernel.
            If left unspecified, it will be tuned automatically.
        num_blocks: Int. The number of conv blocks, each of which may contain
            convolutional, max pooling, dropout, and activation. If left unspecified,
            it will be tuned automatically.
        num_layers: Int. The number of convolutional layers in each block. If left
            unspecified, it will be tuned automatically.
        max_pooling: Boolean. Whether to use max pooling layer in each block. If left
            unspecified, it will be tuned automatically.
        separable: Boolean. Whether to use separable conv layers.
            If left unspecified, it will be tuned automatically.
        dropout: Float. Between 0 and 1. The dropout rate for after the
            convolutional layers. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(
        self,
        kernel_size: Optional[Union[int, hyperparameters.Choice]] = None,
        num_blocks: Optional[int] = None,
        num_layers: Optional[int] = None,
        max_pooling: Optional[bool] = None,
        separable: Optional[bool] = None,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_size = utils.get_hyperparameter(
            kernel_size,
            hyperparameters.Choice("kernel_size", [3, 5, 7], default=3),
            int,
        )
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.max_pooling = max_pooling
        self.separable = separable
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": hyperparameters.serialize(self.kernel_size),
                "num_blocks": self.num_blocks,
                "num_layers": self.num_layers,
                "max_pooling": self.max_pooling,
                "separable": self.separable,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["kernel_size"] = hyperparameters.deserialize(config["kernel_size"])
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        kernel_size = utils.add_to_hp(self.kernel_size, hp)
        num_blocks = self.num_blocks or hp.Choice("num_blocks", [1, 2, 3], default=2)
        num_layers = self.num_layers or hp.Choice("num_layers", [1, 2], default=2)
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

        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)

        for i in range(num_blocks):
            for j in range(num_layers):
                output_node = conv(
                    hp.Choice(
                        "filters_{i}_{j}".format(i=i, j=j),
                        [16, 32, 64, 128, 256, 512],
                        default=32,
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
            if dropout > 0:
                output_node = layers.Dropout(dropout)(output_node)
        return output_node

    @staticmethod
    def _get_padding(kernel_size, output_node):
        if all([kernel_size * 2 <= length for length in output_node.shape[1:-1]]):
            return "valid"
        return "same"


class MultiHeadSelfAttention(block_module.Block):
    """Block for Multi-Head Self-Attention.

    # Arguments
        head_size: Int. Dimensionality of the `query`, `key` and `value` tensors
            after the linear transformation. If left unspecified, it will be
            tuned automatically.
        num_heads: Int. The number of attention heads. Defaults to 8.
    """

    def __init__(
        self, head_size: Optional[int] = None, num_heads: int = 8, **kwargs
    ):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads

    def get_config(self):
        config = super().get_config()
        config.update({"head_size": self.head_size, "num_heads": self.num_heads})
        return config

    def build(self, hp, inputs=None):
        """
        # Arguments
             hp: HyperParameters. The hyperparameters for building the model.
             inputs: Tensor of Shape [batch_size, seq_len, embedding_dim]

        # Returns
            Self-Attention outputs of shape `[batch_size, seq_len, embedding_dim]`.
        """
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        num_heads = self.num_heads
        head_size = (
            self.head_size
            or hp.Choice("head_size_factor", [4, 8, 16, 32, 64], default=16)
            * num_heads
        )

        projection_dim = head_size // num_heads
        query_dense = layers.Dense(head_size)
        key_dense = layers.Dense(head_size)
        value_dense = layers.Dense(head_size)
        combine_heads = layers.Dense(head_size)
        batch_size = tf.shape(input_node)[0]
        query = query_dense(input_node)  # (batch_size, seq_len, head_size)
        key = key_dense(input_node)  # (batch_size, seq_len, head_size)
        value = value_dense(input_node)  # (batch_size, seq_len, head_size)
        query, key, value = [
            self.separate_heads(var, batch_size, num_heads, projection_dim)
            for var in [query, key, value]
        ]
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, tf.shape(attention)[1], self.head_size)
        )  # (batch_size, seq_len, head_size)
        output = combine_heads(concat_attention)  # (batch_size, seq_len, head_size)
        return output

    @staticmethod
    def attention(query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    @staticmethod
    def separate_heads(x, batch_size, num_heads, projection_dim):
        x = tf.reshape(x, (batch_size, -1, num_heads, projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])


class Transformer(block_module.Block):
    """Block for Transformer.
    The input should be tokenized sequences with the same length, where each element
    of a sequence should be the index of the word. The implementation is derived from
    the this
    [example](https://keras.io/examples/nlp/text_classification_with_transformer/).

    # Example
    ```python
        # Using the Transformer Block with AutoModel.
        import autokeras as ak
        from tensorflow.keras import losses
        text_input = ak.TextInput()
        output_node = ak.TextToIntSequence(output_sequence_length=200)(text_input)
        output_node = ak.Transformer(embedding_dim=32,
                             pretraining='none',
                             num_heads=2,
                             dense_dim=32,
                             dropout = 0.25)(output_node)
        output_node = ak.SpatialReduction(reduction_type='global_avg')(output_node)
        output_node = ak.DenseBlock(num_layers=1, use_batchnorm = False)(output_node)
        output_node = ak.ClassificationHead(
            loss=losses.SparseCategoricalCrossentropy(),
            dropout = 0.25)(output_node)
        clf = ak.AutoModel(inputs=text_input, outputs=output_node, max_trials=2)
    ```
    # Arguments
        max_features: Int. Size of the vocabulary. Must be set if not using
            TextToIntSequence before this block. Defaults to 20001.
        pretraining: String. 'random' (use random weights instead any pretrained
            model), 'glove', 'fasttext' or 'word2vec'. Use pretrained word embedding.
            If left unspecified, it will be tuned automatically.
        embedding_dim: Int. Output dimension of the Attention block.
            If left unspecified, it will be tuned automatically.
        num_heads: Int. The number of attention heads. If left unspecified,
            it will be tuned automatically.
        dense_dim: Int. The output dimension of the Feed-Forward Network. If left
            unspecified, it will be tuned automatically.
        dropout: Float. Between 0 and 1. If left unspecified, it will be
            tuned automatically.
    """

    def __init__(
        self,
        max_features: int = 20001,
        pretraining: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        dense_dim: Optional[int] = None,
        dropout: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_features = max_features
        self.pretraining = pretraining
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_features": self.max_features,
                "pretraining": self.pretraining,
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "dense_dim": self.dense_dim,
                "dropout": self.dropout,
            }
        )
        return config

    def build(self, hp, inputs=None):
        """
        # Arguments
             hp: HyperParameters. The hyperparameters for building the model.
             inputs: Tensor of Shape [batch_size, seq_len]

        # Returns
            Output Tensor of shape `[batch_size, seq_len, embedding_dim]`.
        """
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        pretraining = self.pretraining or hp.Choice(
            "pretraining",
            ["random", "glove", "fasttext", "word2vec", "none"],
            default="none",
        )
        embedding_dim = self.embedding_dim or hp.Choice(
            "embedding_dim", [32, 64, 128, 256, 512], default=128
        )
        num_heads = self.num_heads or hp.Choice("num_heads", [8, 16, 32], default=8)

        dense_dim = self.dense_dim or hp.Choice(
            "dense_dim", [128, 256, 512, 1024, 2048], default=2048
        )
        dropout = self.dropout or hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)

        ffn = tf.keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embedding_dim),
            ]
        )

        layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        dropout1 = layers.Dropout(dropout)
        dropout2 = layers.Dropout(dropout)
        # Token and Position Embeddings
        input_node = nest.flatten(inputs)[0]
        token_embedding = Embedding(
            max_features=self.max_features,
            pretraining=pretraining,
            embedding_dim=embedding_dim,
            dropout=dropout,
        ).build(hp, input_node)
        maxlen = input_node.shape[-1]
        batch_size = tf.shape(input_node)[0]
        positions = self.pos_array_funct(maxlen, batch_size)
        position_embedding = Embedding(
            max_features=maxlen,
            pretraining=pretraining,
            embedding_dim=embedding_dim,
            dropout=dropout,
        ).build(hp, positions)
        output_node = tf.keras.layers.Add()([token_embedding, position_embedding])
        attn_output = MultiHeadSelfAttention(embedding_dim, num_heads).build(
            hp, output_node
        )
        attn_output = dropout1(attn_output)
        add_inputs_1 = tf.keras.layers.Add()([output_node, attn_output])
        out1 = layernorm1(add_inputs_1)
        ffn_output = ffn(out1)
        ffn_output = dropout2(ffn_output)
        add_inputs_2 = tf.keras.layers.Add()([out1, ffn_output])
        output = layernorm2(add_inputs_2)
        return output

    @staticmethod
    def pos_array_funct(maxlen, batch_size):
        pos_ones = tf.ones((batch_size, 1), dtype=tf.int32)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = tf.expand_dims(positions, 0)
        positions = tf.matmul(pos_ones, positions)
        return positions


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
        input_node = nest.flatten(inputs)[0]

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
            input_node = layers.experimental.preprocessing.Resizing(
                max(min_size, input_node.shape[1]),
                max(min_size, input_node.shape[2]),
            )(input_node)
        if input_node.shape[3] == 1:
            input_node = layers.Concatenate()([input_node] * 3)
        if input_node.shape[3] != 3:
            input_node = layers.Conv2D(filters=3, kernel_size=1, padding="same")(
                input_node
            )

        if pretrained:
            model = self.models[version](weights="imagenet", include_top=False)
            model.trainable = trainable
        else:
            model = self.models[version](
                weights=None, include_top=False, input_shape=input_node.shape[1:]
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
        super().__init__(pretrained=pretrained, models=models, min_size=32, **kwargs)
        self.version = version

    def get_config(self):
        config = super().get_config()
        config.update({"version": self.version})
        return config


class XceptionBlock(KerasApplicationBlock):
    """Block for XceptionNet.

    An Xception structure, used for specifying your model with specific datasets.

    The original Xception architecture is from https://arxiv.org/abs/1610.02357.
    The data first goes through the entry flow, then through the middle flow which
    is repeated eight times, and finally through the exit flow.

    This XceptionBlock returns a similar architecture as Xception except without
    the last (optional) fully connected layer(s) and logistic regression.
    The size of this architecture could be decided by `HyperParameters`, to get an
    architecture with a half, an identical, or a double size of the original one.

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
        version: String. The value should be one of 'b0', 'b1', ..., 'b7'.
            The type of EfficientNet to use. If left unspecified, it will be tuned
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


class Embedding(block_module.Block):
    """Word embedding block for sequences.

    The input should be tokenized sequences with the same length, where each element
    of a sequence should be the index of the word.

    # Arguments
        max_features: Int. Size of the vocabulary. Must be set if not using
            TextToIntSequence before this block. Defaults to 20001.
        pretraining: String. 'random' (use random weights instead any pretrained
            model), 'glove', 'fasttext' or 'word2vec'. Use pretrained word embedding.
            If left unspecified, it will be tuned automatically.
        embedding_dim: Int. If left unspecified, it will be tuned automatically.
        dropout: Float. The dropout rate for after the Embedding layer.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        max_features: int = 20001,
        pretraining: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_features = max_features
        self.pretraining = pretraining
        self.embedding_dim = embedding_dim
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_features": self.max_features,
                "pretraining": self.pretraining,
                "embedding_dim": self.embedding_dim,
                "dropout": self.dropout,
            }
        )
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        # TODO: support more pretrained embedding layers.
        # glove, fasttext, and word2vec
        pretraining = self.pretraining or hp.Choice(
            "pretraining",
            ["random", "glove", "fasttext", "word2vec", "none"],
            default="none",
        )
        embedding_dim = self.embedding_dim or hp.Choice(
            "embedding_dim", [32, 64, 128, 256, 512], default=128
        )
        if pretraining != "none":
            # TODO: load from pretrained weights
            layer = layers.Embedding(
                input_dim=self.max_features,
                output_dim=embedding_dim,
                input_length=input_node.shape[1],
            )
            # trainable=False,
            # weights=[embedding_matrix])
        else:
            layer = layers.Embedding(
                input_dim=self.max_features, output_dim=embedding_dim
            )
            # input_length=input_node.shape[1],
            # trainable=True)
        output_node = layer(input_node)
        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0.25)
        if dropout > 0:
            output_node = layers.Dropout(dropout)(output_node)
        return output_node


class BertBlock(block_module.Block):
    """Block for Pre-trained BERT.
    The input should be sequence of sentences. The implementation is derived from
    this [example](https://www.tensorflow.org/official_models/fine_tuning_bert)

    # Example
    ```python
        # Using the Transformer Block with AutoModel.
        import autokeras as ak
        from autokeras import BertBlock
        from tensorflow.keras import losses

        input_node = ak.TextInput()
        output_node = BertBlock(max_sequence_length=128)(input_node)
        output_node = ak.ClassificationHead()(output_node)
        clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
    ```
    # Arguments
        max_sequence_length: Int. The maximum length of a sequence that is
            used to train the model.
    """

    def __init__(
        self,
        max_sequence_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_sequence_length = max_sequence_length

    def get_config(self):
        config = super().get_config()
        config.update({"max_sequence_length": self.max_sequence_length})
        return config

    def build(self, hp, inputs=None):
        input_tensor = nest.flatten(inputs)[0]

        max_sequence_length = self.max_sequence_length or hp.Choice(
            "max_seq_len", [128, 256, 512], default=128
        )

        tokenizer_layer = keras_layers.BertTokenizer(
            max_sequence_length=max_sequence_length
        )
        output_node = tokenizer_layer(input_tensor)

        bert_encoder = keras_layers.BertEncoder()

        output_node = bert_encoder(output_node)
        bert_encoder.load_pretrained_weights()

        return output_node
