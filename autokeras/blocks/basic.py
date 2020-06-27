from typing import Optional

import tensorflow as tf
from kerastuner.applications import resnet
from kerastuner.applications import xception
from tensorflow.keras import layers
from tensorflow.python.util import nest

from autokeras.blocks import reduction
from autokeras.engine import block as block_module
from autokeras.utils import layer_utils
from autokeras.utils import utils


def set_hp_value(hp, name, value):
    full_name = hp._get_name(name)
    hp.values[full_name] = value or hp.values[full_name]


class DenseBlock(block_module.Block):
    """Block for Dense layers.

    # Arguments
        num_layers: Int. The number of Dense layers in the block.
            If left unspecified, it will be tuned automatically.
        use_bn: Boolean. Whether to use BatchNormalization layers.
            If left unspecified, it will be tuned automatically.
        dropout_rate: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 num_layers: Optional[int] = None,
                 use_batchnorm: Optional[bool] = None,
                 dropout_rate: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'use_batchnorm': self.use_batchnorm,
            'dropout_rate': self.dropout_rate})
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        output_node = reduction.Flatten().build(hp, output_node)

        num_layers = self.num_layers or hp.Choice('num_layers', [1, 2, 3], default=2)
        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Boolean('use_batchnorm', default=False)
        if self.dropout_rate is not None:
            dropout_rate = self.dropout_rate
        else:
            dropout_rate = hp.Choice('dropout_rate', [0.0, 0.25, 0.5], default=0)

        for i in range(num_layers):
            units = hp.Choice(
                'units_{i}'.format(i=i),
                [16, 32, 64, 128, 256, 512, 1024],
                default=32)
            output_node = layers.Dense(units)(output_node)
            if use_batchnorm:
                output_node = layers.BatchNormalization()(output_node)
            output_node = layers.ReLU()(output_node)
            if dropout_rate > 0:
                output_node = layers.Dropout(dropout_rate)(output_node)
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

    def __init__(self,
                 return_sequences: bool = False,
                 bidirectional: Optional[bool] = None,
                 num_layers: Optional[int] = None,
                 layer_type: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.layer_type = layer_type

    def get_config(self):
        config = super().get_config()
        config.update({
            'return_sequences': self.return_sequences,
            'bidirectional': self.bidirectional,
            'num_layers': self.num_layers,
            'layer_type': self.layer_type})
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        shape = input_node.shape.as_list()
        if len(shape) != 3:
            raise ValueError(
                'Expect the input tensor to have '
                'at least 3 dimensions for rnn models, '
                'but got {shape}'.format(shape=input_node.shape))

        feature_size = shape[-1]
        output_node = input_node

        bidirectional = self.bidirectional
        if bidirectional is None:
            bidirectional = hp.Boolean('bidirectional', default=True)
        layer_type = self.layer_type or hp.Choice('layer_type',
                                                  ['gru', 'lstm'],
                                                  default='lstm')
        num_layers = self.num_layers or hp.Choice('num_layers',
                                                  [1, 2, 3],
                                                  default=2)
        rnn_layers = {
            'gru': layers.GRU,
            'lstm': layers.LSTM
        }
        in_layer = rnn_layers[layer_type]
        for i in range(num_layers):
            return_sequences = True
            if i == num_layers - 1:
                return_sequences = self.return_sequences
            if bidirectional:
                output_node = layers.Bidirectional(
                    in_layer(feature_size,
                             return_sequences=return_sequences))(output_node)
            else:
                output_node = in_layer(
                    feature_size,
                    return_sequences=return_sequences)(output_node)
        return output_node


class ConvBlock(block_module.Block):
    """Block for vanilla ConvNets.

    # Arguments
        kernel_size: Int. If left unspecified, it will be tuned automatically.
        num_blocks: Int. The number of conv blocks, each of which may contain
            convolutional, max pooling, dropout, and activation. If left unspecified,
            it will be tuned automatically.
        num_layers: Int. The number of convolutional layers in each block. If left
            unspecified, it will be tuned automatically.
        max_pooling: Boolean. Whether to use max pooling layer in each block. If left
            unspecified, it will be tuned automatically.
        separable: Boolean. Whether to use separable conv layers.
            If left unspecified, it will be tuned automatically.
        dropout_rate: Float. Between 0 and 1. The dropout rate for after the
            convolutional layers. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(self,
                 kernel_size: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 num_layers: Optional[int] = None,
                 max_pooling: Optional[bool] = None,
                 separable: Optional[bool] = None,
                 dropout_rate: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.max_pooling = max_pooling
        self.separable = separable
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'num_blocks': self.num_blocks,
            'num_layers': self.num_layers,
            'max_pooling': self.max_pooling,
            'separable': self.separable,
            'dropout_rate': self.dropout_rate})
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        kernel_size = self.kernel_size or hp.Choice('kernel_size',
                                                    [3, 5, 7],
                                                    default=3)
        num_blocks = self.num_blocks or hp.Choice('num_blocks',
                                                  [1, 2, 3],
                                                  default=2)
        num_layers = self.num_layers or hp.Choice('num_layers',
                                                  [1, 2],
                                                  default=2)
        separable = self.separable
        if separable is None:
            separable = hp.Boolean('separable', default=False)

        if separable:
            conv = layer_utils.get_sep_conv(input_node.shape)
        else:
            conv = layer_utils.get_conv(input_node.shape)

        max_pooling = self.max_pooling
        if max_pooling is None:
            max_pooling = hp.Boolean('max_pooling', default=True)
        pool = layer_utils.get_max_pooling(input_node.shape)

        if self.dropout_rate is not None:
            dropout_rate = self.dropout_rate
        else:
            dropout_rate = hp.Choice('dropout_rate', [0.0, 0.25, 0.5], default=0)

        for i in range(num_blocks):
            for j in range(num_layers):
                output_node = conv(
                    hp.Choice('filters_{i}_{j}'.format(i=i, j=j),
                              [16, 32, 64, 128, 256, 512],
                              default=32),
                    kernel_size,
                    padding=self._get_padding(kernel_size, output_node),
                    activation='relu')(output_node)
            if max_pooling:
                output_node = pool(
                    kernel_size - 1,
                    padding=self._get_padding(kernel_size - 1,
                                              output_node))(output_node)
            if dropout_rate > 0:
                output_node = layers.Dropout(dropout_rate)(output_node)
        return output_node

    @staticmethod
    def _get_padding(kernel_size, output_node):
        if all([kernel_size * 2 <= length
                for length in output_node.shape[1:-1]]):
            return 'valid'
        return 'same'


class MultiHeadSelfAttentionBlock(block_module.Block):
    """Block for Multi-Head Self-Attention.

    # Arguments
        embed_dim: Int. Output dimension of the Attention block.
            If left unspecified, it will be tuned automatically.
        num_heads: Int. The number of attention heads. If left unspecified,
            it will be tuned automatically.
    """

    def __init__(self,
                 embed_dim: Optional[int] = None,
                 num_heads: Optional[int] = 8,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads})
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
        shape = input_node.shape.as_list()
        if len(shape) != 3:
            raise ValueError(
                'Expect the input tensor to have '
                '3 dimensions for multi-head self-attention, '
                'but got {shape}'.format(shape=input_node.shape))
        # input.shape = [batch_size, seq_len, embedding_dim]
        embed_dim = self.embed_dim or hp.Choice(
            'embed_dim',
            [32, 64, 128, 256, 512],
            default=128)
        num_heads = self.num_heads
        if num_heads is None:
            num_heads = 8

        if embed_dim % num_heads != 0:  # how to evaluate this condition
            raise ValueError(
                f"embedding dimension = {embed_dim} should be "
                f"divisible by number of heads = {num_heads}"
            )
        projection_dim = embed_dim // num_heads
        query_dense = layers.Dense(embed_dim)
        key_dense = layers.Dense(embed_dim)
        value_dense = layers.Dense(embed_dim)
        combine_heads = layers.Dense(embed_dim)
        print("MultiHeadSelfAttentionBlock: ", shape)
        batch_size = tf.shape(input_node)[0]
        query = query_dense(input_node)  # (batch_size, seq_len, embed_dim)
        key = key_dense(input_node)  # (batch_size, seq_len, embed_dim)
        value = value_dense(input_node)  # (batch_size, seq_len, embed_dim)
        query, key, value = [self.separate_heads(
            var, batch_size, num_heads,  projection_dim
        ) for var in [query, key, value]]
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
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


class TransformerBlock(block_module.Block):
    """Block for Transformer.

    # Arguments
    embed_dim: Int. Output dimension of the Attention block.
        If left unspecified, it will be tuned automatically.
    num_heads: Int. The number of attention heads. If left unspecified,
        it will be tuned automatically.
    ff_dim: Int. The output dimension of the FFN. If left
        unspecified, it will be tuned automatically.
    dropout_rate: Float. Between 0 and 1. If left unspecified, it will be
        tuned automatically.
    """

    def __init__(self,
                 embed_dim: Optional[int] = None,
                 num_heads: Optional[int] = None,
                 ff_dim: Optional[int] = None,
                 dropout_rate: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self. ff_dim = ff_dim
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate})
        return config

    def build(self, hp, inputs=None):
        """
        # Arguments
             hp: HyperParameters. The hyperparameters for building the model.
             inputs: Tensor of Shape [batch_size, seq_len, embedding_dim]

        # Returns
            Output Tensor of shape `[batch_size, seq_len, embedding_dim]`.
        """
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        embed_dim = self.embed_dim or hp.Choice(
            'embed_dim',
            [32, 64, 128, 256, 512],
            default=128)
        num_heads = self.num_heads or hp.Choice('num_heads', [8, 16, 32], default=8)

        ff_dim = self.ff_dim or hp.Choice('ff_dim',
                                          [128, 256, 512, 1024, 2048],
                                          default=2048)
        dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
                                                      [0.0, 0.25, 0.5],
                                                      default=0)

        ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        dropout1 = layers.Dropout(dropout_rate)
        dropout2 = layers.Dropout(dropout_rate)
        # print("TransformerBlock ", inputs.shape)
        attn_output = MultiHeadSelfAttentionBlock(
            embed_dim, num_heads).build(hp, inputs)
        attn_output = dropout1(attn_output)
        out1 = layernorm1(inputs + attn_output)
        ffn_output = ffn(out1)
        ffn_output = dropout2(ffn_output)
        output = layernorm2(out1 + ffn_output)
        return output


class ResNetBlock(resnet.HyperResNet, block_module.Block):
    """Block for ResNet.

    # Arguments
        version: String. 'v1', 'v2' or 'next'. The type of ResNet to use.
            If left unspecified, it will be tuned automatically.
        pooling: String. 'avg', 'max'. The type of pooling layer to use.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 version: Optional[str] = None,
                 pooling: Optional[str] = None,
                 **kwargs):
        if 'include_top' in kwargs:
            raise ValueError(
                'Argument "include_top" is not supported in ResNetBlock.')
        if 'input_shape' in kwargs:
            raise ValueError(
                'Argument "input_shape" is not supported in ResNetBlock.')
        super().__init__(include_top=False, input_shape=(10,), **kwargs)
        self.version = version
        self.pooling = pooling

    def get_config(self):
        config = super().get_config()
        config.update({
            'version': self.version,
            'pooling': self.pooling})
        return config

    def build(self, hp, inputs=None):
        self.input_tensor = nest.flatten(inputs)[0]
        self.input_shape = None

        hp.Choice('version', ['v1', 'v2', 'next'], default='v2')
        hp.Choice('pooling', ['avg', 'max'], default='avg')

        set_hp_value(hp, 'version', self.version)
        set_hp_value(hp, 'pooling', self.pooling)

        model = super().build(hp)
        return model.outputs


class XceptionBlock(xception.HyperXception, block_module.Block):
    """XceptionBlock.

    An Xception structure, used for specifying your model with specific datasets.

    The original Xception architecture is from https://arxiv.org/abs/1610.02357.
    The data first goes through the entry flow, then through the middle flow which
    is repeated eight times, and finally through the exit flow.

    This XceptionBlock returns a similar architecture as Xception except without
    the last (optional) fully connected layer(s) and logistic regression.
    The size of this architecture could be decided by `HyperParameters`, to get an
    architecture with a half, an identical, or a double size of the original one.

    # Arguments
        activation: String. 'selu' or 'relu'. If left unspecified, it will be tuned
            automatically.
        initial_strides: Int. If left unspecified, it will be tuned automatically.
        num_residual_blocks: Int. If left unspecified, it will be tuned
            automatically.
        pooling: String. 'ave', 'flatten', or 'max'. If left unspecified, it will be
            tuned automatically.
    """

    def __init__(self,
                 activation: Optional[str] = None,
                 initial_strides: Optional[int] = None,
                 num_residual_blocks: Optional[int] = None,
                 pooling: Optional[str] = None,
                 **kwargs):
        if 'include_top' in kwargs:
            raise ValueError(
                'Argument "include_top" is not supported in XceptionBlock.')
        if 'input_shape' in kwargs:
            raise ValueError(
                'Argument "input_shape" is not supported in XceptionBlock.')
        super().__init__(include_top=False, input_shape=(10,), **kwargs)
        self.activation = activation
        self.initial_strides = initial_strides
        self.num_residual_blocks = num_residual_blocks
        self.pooling = pooling

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'activation': self.activation,
            'initial_strides': self.initial_strides,
            'num_residual_blocks': self.num_residual_blocks,
            'pooling': self.pooling})
        return config

    def build(self, hp, inputs=None):
        self.input_tensor = nest.flatten(inputs)[0]
        self.input_shape = None

        hp.Choice('activation', ['relu', 'selu'])
        hp.Choice('initial_strides', [2])
        hp.Int('num_residual_blocks', 2, 8, default=4)
        hp.Choice('pooling', ['avg', 'flatten', 'max'])

        set_hp_value(hp, 'activation', self.activation)
        set_hp_value(hp, 'initial_strides', self.initial_strides)
        set_hp_value(hp, 'num_residual_blocks', self.num_residual_blocks)
        set_hp_value(hp, 'pooling', self.pooling)

        model = super().build(hp)
        return model.outputs


class TokenAndPositionEmbedding(block_module.Block):
    """Token and Position Embedding block for Transformer.

    The input should be tokenized sequences with the same length, where each element
    of a sequence should be the index of the word.

    # Arguments
        max_features: Int. Size of the vocabulary. Must be set if not using
            TextToIntSequence before this block. Defaults to 20001.
        pretraining: String. 'random' (use random weights instead any pretrained
            model), 'glove', 'fasttext' or 'word2vec'. Use pretrained word embedding.
            If left unspecified, it will be tuned automatically.
        embedding_dim: Int. If left unspecified, it will be tuned automatically.
        dropout_rate: Float. The dropout rate for after the Embedding layer.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 max_features: int = 20001,
                 pretraining: Optional[str] = None,
                 embedding_dim: Optional[int] = None,
                 dropout_rate: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_features = max_features
        self.pretraining = pretraining
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_features': self.max_features,
            'pretraining': self.pretraining,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate})
        return config

    def build(self, hp, inputs=None):
        """
        # Arguments
             hp: HyperParameters. The hyperparameters for building the model.
             inputs: Tensor of Shape [batch_size, seq_len]

        # Returns
            Output Tensor of shape `[batch_size, seq_len, embedding_dim]`.
        """
        input_node = nest.flatten(inputs)[0]
        pretraining = self.pretraining or hp.Choice(
            'pretraining',
            ['random', 'glove', 'fasttext', 'word2vec', 'none'],
            default='none')
        embedding_dim = self.embedding_dim or hp.Choice(
            'embedding_dim',
            [32, 64, 128, 256, 512],
            default=128)

        if self.dropout_rate is not None:
            dropout_rate = self.dropout_rate
        else:
            dropout_rate = hp.Choice('dropout_rate', [0.0, 0.25, 0.5], default=0.25)

        token_embedding = Embedding(max_features=self.max_features,
                                    pretraining=pretraining,
                                    embedding_dim=embedding_dim,
                                    dropout_rate=dropout_rate).build(hp, input_node)
        print("TokenAndPositionEmbedding ", input_node.shape)
        maxlen = tf.shape(input_node)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embedding = Embedding(max_features=maxlen,
                                       pretraining=pretraining,
                                       embedding_dim=embedding_dim,
                                       dropout_rate=dropout_rate).build(hp, positions)

        output_node = tf.keras.layers.Add()([token_embedding,
                                             position_embedding])

        # if pretraining != 'none':
        #     # TODO: load from pretrained weights
        #     poition_layer = layers.Embedding(
        #         input_dim=self.max_features,
        #         output_dim=embedding_dim,
        #         input_length=input_node.shape[1])
        #     # trainable=False,
        #     # weights=[embedding_matrix])
        # else:
        #     layer = layers.Embedding(
        #         input_dim=self.max_features,
        #         output_dim=embedding_dim)
        #     # input_length=input_node.shape[1],
        #     # trainable=True)
        # output_node = layer(input_node)
        # if self.dropout_rate is not None:
        #     dropout_rate = self.dropout_rate
        # else:
        #     dropout_rate = hp.Choice('dropout_rate', [0.0, 0.25, 0.5], default=0.25)
        # if dropout_rate > 0:
        #     output_node = layers.Dropout(dropout_rate)(output_node)

        return output_node


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
        dropout_rate: Float. The dropout rate for after the Embedding layer.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 max_features: int = 20001,
                 pretraining: Optional[str] = None,
                 embedding_dim: Optional[int] = None,
                 dropout_rate: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_features = max_features
        self.pretraining = pretraining
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_features': self.max_features,
            'pretraining': self.pretraining,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate})
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        # TODO: support more pretrained embedding layers.
        # glove, fasttext, and word2vec
        pretraining = self.pretraining or hp.Choice(
            'pretraining',
            ['random', 'glove', 'fasttext', 'word2vec', 'none'],
            default='none')
        embedding_dim = self.embedding_dim or hp.Choice(
            'embedding_dim',
            [32, 64, 128, 256, 512],
            default=128)
        if pretraining != 'none':
            # TODO: load from pretrained weights
            layer = layers.Embedding(
                input_dim=self.max_features,
                output_dim=embedding_dim,
                input_length=input_node.shape[1])
            # trainable=False,
            # weights=[embedding_matrix])
        else:
            layer = layers.Embedding(
                input_dim=self.max_features,
                output_dim=embedding_dim)
            # input_length=input_node.shape[1],
            # trainable=True)
        output_node = layer(input_node)
        if self.dropout_rate is not None:
            dropout_rate = self.dropout_rate
        else:
            dropout_rate = hp.Choice('dropout_rate', [0.0, 0.25, 0.5], default=0.25)
        if dropout_rate > 0:
            output_node = layers.Dropout(dropout_rate)(output_node)
        return output_node
