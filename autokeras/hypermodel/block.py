import tensorflow as tf
from kerastuner.applications import resnet
from kerastuner.applications import xception
from tensorflow.python.util import nest

from autokeras import utils
from autokeras.hypermodel import base


def set_hp_value(hp, name, value):
    full_name = hp._get_name(name)
    hp.values[full_name] = value or hp.values[full_name]


class DenseBlock(base.Block):
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
                 num_layers=None,
                 use_batchnorm=None,
                 dropout_rate=None,
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

    def set_config(self, config):
        super().set_config(config)
        self.num_layers = config['num_layers']
        self.use_batchnorm = config['use_batchnorm']
        self.dropout_rate = config['dropout_rate']

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        output_node = Flatten().build(hp, output_node)

        num_layers = self.num_layers or hp.Choice('num_layers', [1, 2, 3], default=2)
        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Choice('use_batchnorm', [True, False], default=False)
        dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
                                                      [0.0, 0.25, 0.5],
                                                      default=0)

        for i in range(num_layers):
            units = hp.Choice(
                'units_{i}'.format(i=i),
                [16, 32, 64, 128, 256, 512, 1024],
                default=32)
            output_node = tf.keras.layers.Dense(units)(output_node)
            if use_batchnorm:
                output_node = tf.keras.layers.BatchNormalization()(output_node)
            output_node = tf.keras.layers.ReLU()(output_node)
            output_node = tf.keras.layers.Dropout(dropout_rate)(output_node)
        return output_node


class RNNBlock(base.Block):
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
                 return_sequences=False,
                 bidirectional=None,
                 num_layers=None,
                 layer_type=None,
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

    def set_config(self, config):
        super().set_config(config)
        self.return_sequences = config['return_sequences']
        self.bidirectional = config['bidirectional']
        self.num_layers = config['num_layers']
        self.layer_type = config['layer_type']

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
            bidirectional = hp.Choice('bidirectional', [True, False], default=True)
        layer_type = self.layer_type or hp.Choice('layer_type',
                                                  ['gru', 'lstm'],
                                                  default='lstm')
        num_layers = self.num_layers or hp.Choice('num_layers',
                                                  [1, 2, 3],
                                                  default=2)
        rnn_layers = {
            'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM
        }
        in_layer = rnn_layers[layer_type]
        for i in range(num_layers):
            return_sequences = True
            if i == num_layers - 1:
                return_sequences = self.return_sequences
            if bidirectional:
                output_node = tf.keras.layers.Bidirectional(
                    in_layer(feature_size,
                             return_sequences=return_sequences))(output_node)
            else:
                output_node = in_layer(
                    feature_size,
                    return_sequences=return_sequences)(output_node)
        return output_node


class ConvBlock(base.Block):
    """Block for vanilla ConvNets.

    # Arguments
        kernel_size: Int. If left unspecified, it will be tuned automatically.
        num_blocks: Int. The number of conv blocks. If left unspecified, it will be
            tuned automatically.
        separable: Boolean. Whether to use separable conv layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 kernel_size=None,
                 num_blocks=None,
                 separable=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.separable = separable

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'num_blocks': self.num_blocks,
            'separable': self.separable})
        return config

    def set_config(self, config):
        super().set_config(config)
        self.kernel_size = config['kernel_size']
        self.num_blocks = config['num_blocks']
        self.separable = config['separable']

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
        separable = self.separable
        if separable is None:
            separable = hp.Choice('separable', [True, False], default=False)

        if separable:
            conv = utils.get_sep_conv(input_node.shape)
        else:
            conv = utils.get_conv(input_node.shape)
        pool = utils.get_max_pooling(input_node.shape)

        for i in range(num_blocks):
            output_node = conv(
                hp.Choice('filters_{i}_1'.format(i=i),
                          [16, 32, 64],
                          default=32),
                kernel_size,
                padding=self._get_padding(kernel_size, output_node),
                activation='relu')(output_node)
            output_node = conv(
                hp.Choice('filters_{i}_2'.format(i=i),
                          [16, 32, 64],
                          default=32),
                kernel_size,
                padding=self._get_padding(kernel_size, output_node),
                activation='relu')(output_node)
            output_node = pool(
                kernel_size - 1,
                padding=self._get_padding(kernel_size - 1, output_node))(output_node)
        return output_node

    @staticmethod
    def _get_padding(kernel_size, output_node):
        if (kernel_size * 2 <= output_node.shape[1] and
                kernel_size * 2 <= output_node.shape[2]):
            return 'valid'
        return 'same'


class ResNetBlock(base.Block, resnet.HyperResNet):
    """Block for ResNet.

    # Arguments
        version: String. 'v1', 'v2' or 'next'. The type of ResNet to use.
            If left unspecified, it will be tuned automatically.
        pooling: String. 'avg', 'max'. The type of pooling layer to use.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 version=None,
                 pooling=None,
                 **kwargs):
        super().__init__(include_top=False, input_shape=(10,), **kwargs)
        self.version = version
        self.pooling = pooling

    def get_config(self):
        config = super().get_config()
        config.update({
            'version': self.version,
            'pooling': self.pooling})
        return config

    def set_config(self, config):
        super().set_config(config)
        self.version = config['version']
        self.pooling = config['pooling']

    def build(self, hp, inputs=None):
        self.input_tensor = nest.flatten(inputs)[0]
        self.input_shape = None

        hp.Choice('version', ['v1', 'v2', 'next'], default='v2')
        hp.Choice('pooling', ['avg', 'max'], default='avg')

        set_hp_value(hp, 'version', self.version)
        set_hp_value(hp, 'pooling', self.pooling)

        model = super().build(hp)
        return model.outputs


class XceptionBlock(base.Block, xception.HyperXception):
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
                 activation=None,
                 initial_strides=None,
                 num_residual_blocks=None,
                 pooling=None,
                 **kwargs):
        super().__init__(include_top=False, input_shape=(10,), **kwargs)
        self.activation = activation
        self.initial_strides = initial_strides
        self.num_residual_blocks = num_residual_blocks
        self.pooling = pooling

    def get_config(self):
        config = super().get_config()
        config.update({
            'activation': self.activation,
            'initial_strides': self.initial_strides,
            'num_residual_blocks': self.num_residual_blocks,
            'pooling': self.pooling})
        return config

    def set_config(self, config):
        super().set_config(config)
        self.activation = config['activation']
        self.initial_strides = config['initial_strides']
        self.num_residual_blocks = config['num_residual_blocks']
        self.pooling = config['pooling']

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


def shape_compatible(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    # TODO: If they can be the same after passing through any layer,
    #  they are compatible. e.g. (32, 32, 3), (16, 16, 2) are compatible
    return shape1[:-1] == shape2[:-1]


class Merge(base.Block):
    """Merge block to merge multiple nodes into one.

    # Arguments
        merge_type: String. 'add' or 'concatenate'. If left unspecified, it will be
            tuned automatically.
    """

    def __init__(self, merge_type=None, **kwargs):
        super().__init__(**kwargs)
        self.merge_type = merge_type

    def get_config(self):
        config = super().get_config()
        config.update({'merge_type': self.merge_type})
        return config

    def set_config(self, config):
        super().set_config(config)
        self.merge_type = config['merge_type']

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        if len(inputs) == 1:
            return inputs

        merge_type = self.merge_type or hp.Choice('merge_type',
                                                  ['add', 'concatenate'],
                                                  default='add')

        if not all([shape_compatible(input_node.shape, inputs[0].shape) for
                    input_node in inputs]):
            new_inputs = []
            for input_node in inputs:
                new_inputs.append(Flatten().build(hp, input_node))
            inputs = new_inputs

        # TODO: Even inputs have different shape[-1], they can still be Add(
        #  ) after another layer. Check if the inputs are all of the same
        #  shape
        if all([input_node.shape == inputs[0].shape for input_node in inputs]):
            if merge_type == 'add':
                return tf.keras.layers.Add(inputs)

        return tf.keras.layers.Concatenate()(inputs)


class Flatten(base.Block):
    """Flatten the input tensor with Keras Flatten layer."""

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        if len(input_node.shape) > 2:
            return tf.keras.layers.Flatten()(input_node)
        return input_node


class SpatialReduction(base.Block):
    """Reduce the dimension of a spatial tensor, e.g. image, to a vector.

    # Arguments
        reduction_type: String. 'flatten', 'global_max' or 'global_avg'.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, reduction_type=None, **kwargs):
        super().__init__(**kwargs)
        self.reduction_type = reduction_type

    def get_config(self):
        config = super().get_config()
        config.update({'reduction_type': self.reduction_type})
        return config

    def set_config(self, config):
        super().set_config(config)
        self.reduction_type = config['reduction_type']

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        # No need to reduce.
        if len(output_node.shape) <= 2:
            return output_node

        reduction_type = self.reduction_type or hp.Choice('reduction_type',
                                                          ['flatten',
                                                           'global_max',
                                                           'global_avg'],
                                                          default='global_avg')
        if reduction_type == 'flatten':
            output_node = Flatten().build(hp, output_node)
        elif reduction_type == 'global_max':
            output_node = utils.get_global_max_pooling(
                output_node.shape)()(output_node)
        elif reduction_type == 'global_avg':
            output_node = utils.get_global_average_pooling(
                output_node.shape)()(output_node)
        return output_node


class TemporalReduction(base.Block):
    """Reduce the dimension of a temporal tensor, e.g. output of RNN, to a vector.

    # Arguments
        reduction_type: String. 'flatten', 'global_max' or 'global_avg'. If left
            unspecified, it will be tuned automatically.
    """

    def __init__(self, reduction_type=None, **kwargs):
        super().__init__(**kwargs)
        self.reduction_type = reduction_type

    def get_config(self):
        config = super().get_config()
        config.update({'reduction_type': self.reduction_type})
        return config

    def set_config(self, config):
        super().set_config(config)
        self.reduction_type = config['reduction_type']

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        # No need to reduce.
        if len(output_node.shape) <= 2:
            return output_node

        reduction_type = self.reduction_type or hp.Choice('reduction_type',
                                                          ['flatten',
                                                           'global_max',
                                                           'global_avg'],
                                                          default='global_avg')

        if reduction_type == 'flatten':
            output_node = Flatten().build(hp, output_node)
        elif reduction_type == 'global_max':
            output_node = tf.math.reduce_max(output_node, axis=-2)
        elif reduction_type == 'global_avg':
            output_node = tf.math.reduce_mean(output_node, axis=-2)
        elif reduction_type == 'global_min':
            output_node = tf.math.reduce_min(output_node, axis=-2)

        return output_node


class EmbeddingBlock(base.Block):
    """Word embedding block for sequences.

    The input should be tokenized sequences with the same length, where each element
    of a sequence should be the index of the word.

    # Arguments
        max_features: Int. Size of the vocabulary. Must be set if not using
            TextToIntSequence before this block. If not specified, we will use the
            vocabulary size in the preceding TextToIntSequence vocabulary size.
        pretraining: String. 'random' (use random weights instead any pretrained
            model), 'glove', 'fasttext' or 'word2vec'. Use pretrained word embedding.
            If left unspecified, it will be tuned automatically.
        embedding_dim: Int. If left unspecified, it will be tuned automatically.
        dropout_rate: Float. The dropout rate for after the Embedding layer.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 max_features=None,
                 pretraining=None,
                 embedding_dim=None,
                 dropout_rate=None,
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
            'embedding_dim': self.embedding_dim})
        return config

    def set_config(self, config):
        super().set_config(config)
        self.max_features = config['max_features']
        self.pretraining = config['pretraining']
        self.embedding_dim = config['embedding_dim']

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
            layer = tf.keras.layers.Embedding(
                input_dim=self.max_features,
                output_dim=embedding_dim,
                input_length=input_node.shape[1])
            # trainable=False,
            # weights=[embedding_matrix])
        else:
            layer = tf.keras.layers.Embedding(
                input_dim=self.max_features,
                output_dim=embedding_dim,
                input_length=input_node.shape[1],
                trainable=True)
        output_node = layer(input_node)
        dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
                                                      [0.0, 0.25, 0.5],
                                                      default=0.25)
        if dropout_rate > 0:
            output_node = tf.keras.layers.Dropout(dropout_rate)(output_node)
        return output_node
