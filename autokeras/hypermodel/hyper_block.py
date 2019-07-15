import numpy as np
import tensorflow as tf
import kerastuner
from kerastuner.applications import resnet
from kerastuner.applications import xception
from tensorflow.python.util import nest

from autokeras.hypermodel import hyper_node
from autokeras import utils
from autokeras import const


class HyperBlock(kerastuner.HyperModel):

    def __init__(self, **kwargs):
        super(HyperBlock, self).__init__(**kwargs)
        self.inputs = None
        self.outputs = None
        self._num_output_node = 1

    def __call__(self, inputs):
        self.inputs = nest.flatten(inputs)
        for input_node in self.inputs:
            input_node.add_out_hypermodel(self)
        self.outputs = []
        for _ in range(self._num_output_node):
            output_node = hyper_node.Node()
            output_node.add_in_hypermodel(self)
            self.outputs.append(output_node)
        return self.outputs

    def build(self, hp, inputs=None):
        return super(HyperBlock, self).build(hp)


class ResNetBlock(HyperBlock, resnet.HyperResNet):

    def __init__(self, **kwargs):
        super().__init__(include_top=False, input_shape=(10,), **kwargs)

    def build(self, hp, inputs=None):
        self.input_tensor = nest.flatten(inputs)[0]
        self.input_shape = None
        model = super(ResNetBlock, self).build(hp)
        return model.outputs


class DenseBlock(HyperBlock):

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        output_node = Flatten().build(hp, output_node)
        use_bn = hp.Choice(
            'use_bn',
            [True, False],
            default=False)
        dropout_rate = hp.Choice('dropout_rate', [0, 0.25, 0.5], default=0)
        for i in range(hp.Choice('num_layers', [1, 2, 3], default=2)):
            units = hp.Choice(
                'units_{i}'.format(i=i),
                [16, 32, 64, 128, 256, 512, 1024],
                default=32)
            output_node = tf.keras.layers.Dense(units)(output_node)
            if use_bn:
                output_node = tf.keras.layers.BatchNormalization()(output_node)
            output_node = tf.keras.layers.ReLU()(output_node)
            output_node = tf.keras.layers.Dropout(dropout_rate)(output_node)
        return output_node


class RNNBlock(HyperBlock):
    """ An RNN HyperBlock.

    Attributes:
        bidirectional: Boolean. If not provided, it would be a tunable variable.
        return_sequences: Boolean.  If not provided, it would be a tunable variable.
    """

    def __init__(self,
                 bidirectional=None,
                 return_sequences=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences

    @staticmethod
    def attention_block(inputs):
        time_steps = int(inputs.shape[1])
        attention_out = tf.keras.layers.Permute((2, 1))(inputs)
        attention_out = tf.keras.layers.Dense(time_steps,
                                              activation='softmax')(attention_out)
        attention_out = tf.keras.layers.Permute((2, 1))(attention_out)
        mul_attention_out = tf.keras.layers.Multiply()([inputs, attention_out])
        return mul_attention_out
        return inputs

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        shape = input_node.shape.as_list()
        if len(shape) != 3:
            raise ValueError(
                "Expect the input tensor to have "
                "exactly 3 dimensions for rnn models, "
                "but got {shape}".format(shape=input_node.shape))

        feature_size = shape[-1]
        output_node = input_node

        in_layer = const.Constant.RNN_LAYERS[hp.Choice('rnn_type',
                                                       ['gru', 'lstm'],
                                                       default='lstm')]
        choice_of_layers = hp.Choice('num_layers', [1, 2, 3], default=2)

        bidirectional = self.bidirectional
        if bidirectional is None:
            bidirectional = hp.Choice('bidirectional',
                                      [True, False],
                                      default=True)

        return_sequences = self.return_sequences
        if return_sequences is None:
            return_sequences = hp.Choice('return_sequences',
                                         [True, False],
                                         default=True)

        if return_sequences:
            attention_choices = ['pre', 'post', 'none']
        else:
            attention_choices = ['pre', 'none']

        attention_mode = hp.Choice('attention', attention_choices, default='post')
        output_node = self.attention_block(output_node) \
            if attention_mode == 'pre' else output_node

        for i in range(choice_of_layers):
            temp_return_sequences = True
            if i == choice_of_layers - 1:
                temp_return_sequences = return_sequences
            if bidirectional:
                output_node = tf.keras.layers.Bidirectional(
                    in_layer(feature_size,
                             return_sequences=temp_return_sequences))(output_node)
            else:
                output_node = in_layer(
                    feature_size,
                    return_sequences=temp_return_sequences)(output_node)

        output_node = self.attention_block(output_node) \
            if attention_mode == 'post' else output_node

        return output_node


class ImageBlock(HyperBlock):

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        block_type = hp.Choice('block_type',
                               ['resnet', 'xception', 'vanilla'],
                               default='resnet')

        if block_type == 'resnet':
            output_node = ResNetBlock().build(hp, output_node)
        elif block_type == 'xception':
            output_node = XceptionBlock().build(hp, output_node)
        elif block_type == 'vanilla':
            output_node = ConvBlock().build(hp, output_node)
        return output_node


class ConvBlock(HyperBlock):

    def __init__(self, separable=None, **kwargs):
        super().__init__(**kwargs)
        self.separable = separable

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        separable = self.separable
        if separable is None:
            separable = hp.Choice('separable', [True, False], default=False)
        if separable:
            conv = utils.get_sep_conv(input_node.shape)
        else:
            conv = utils.get_conv(input_node.shape)
        pool = utils.get_max_pooling(input_node.shape)
        dropout = utils.get_dropout(input_node.shape)
        kernel_size = hp.Choice('kernel_size', [3, 5, 7], default=3)
        dropout_rate = hp.Choice('dropout_rate', [0, 0.25, 0.5], default=0.5)

        for i in range(hp.Choice('num_blocks', [1, 2, 3], default=2)):
            if dropout_rate > 0:
                output_node = dropout(dropout_rate)(output_node)
            output_node = conv(
                hp.Choice('filters_{i}_1'.format(i=i),
                          [16, 32, 64],
                          default=32),
                kernel_size,
                padding=self._get_padding(kernel_size, output_node))(output_node)
            output_node = conv(
                hp.Choice('filters_{i}_2'.format(i=i),
                          [16, 32, 64],
                          default=32),
                kernel_size,
                padding=self._get_padding(kernel_size, output_node))(output_node)
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


def shape_compatible(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    # TODO: If they can be the same after passing through any layer,
    #  they are compatible. e.g. (32, 32, 3), (16, 16, 2) are compatible
    return shape1[:-1] == shape2[:-1]


class Merge(HyperBlock):

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        if len(inputs) == 1:
            return inputs

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
            if hp.Choice("merge_type", ['Add', 'Concatenate'], default='Add'):
                return tf.keras.layers.Add(inputs)

        return tf.keras.layers.Add()(inputs)


class XceptionBlock(HyperBlock, xception.HyperXception):
    """ XceptionBlock

    An Xception structure, used for specifying your model with specific datasets.

    The original Xception architecture is from https://arxiv.org/abs/1610.02357.
    The data first goes through the entry flow, then through the middle flow which
    is repeated eight times, and finally through the exit flow.

    This XceptionBlock returns a similar architecture as Xception except without
    the last (optional) fully connected layer(s) and logistic regression.
    The size of this architecture could be decided by `HyperParameters`, to get an
    architecture with a half, an identical, or a double size of the original one.

    """

    def __init__(self, **kwargs):
        super().__init__(include_top=False, input_shape=(10,), **kwargs)

    def build(self, hp, inputs=None):
        self.input_tensor = nest.flatten(inputs)[0]
        self.input_shape = None
        model = super(XceptionBlock, self).build(hp)
        return model.outputs


class Flatten(HyperBlock):

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        if len(input_node.shape) > 2:
            return tf.keras.layers.Flatten()(input_node)
        return input_node


class SpatialReduction(HyperBlock):

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        # No need to reduce.
        if len(output_node.shape) <= 2:
            return output_node

        reduction_type = hp.Choice('reduction_type',
                                   ['flatten',
                                    'global_max',
                                    'global_ave'],
                                   default='global_ave')
        if reduction_type == 'flatten':
            output_node = Flatten().build(hp, output_node)
        elif reduction_type == 'global_max':
            output_node = utils.get_global_max_pooling(
                output_node.shape)()(output_node)
        elif reduction_type == 'global_ave':
            output_node = utils.get_global_average_pooling(
                output_node.shape)()(output_node)
        return output_node


class TemporalReduction(HyperBlock):

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        # No need to reduce.
        if len(output_node.shape) <= 2:
            return output_node

        reduction_type = hp.Choice('reduction_type',
                                   ['flatten',
                                    'global_max',
                                    'global_ave',
                                    'global_min'],
                                   default='global_ave')

        if reduction_type == 'flatten':
            output_node = Flatten().build(hp, output_node)
        elif reduction_type == 'global_max':
            output_node = tf.math.reduce_max(output_node, axis=-2)
        elif reduction_type == 'global_ave':
            output_node = tf.math.reduce_mean(output_node, axis=-2)
        elif reduction_type == 'global_min':
            output_node = tf.math.reduce_min(output_node, axis=-2)

        return output_node


class EmbeddingBlock(HyperBlock):
    """Word embedding block for sequences.

    The input should be tokenized sequences with the same length, where each element
    of a sequence should be the index of the word.

    Attributes:
        pretrained: Boolean. Use pretrained word embedding.
    """

    def __init__(self,
                 pretrained=None,
                 is_embedding_trainable=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pretrained = pretrained
        self.is_embedding_trainable = is_embedding_trainable

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        pretrained = self.pretrained
        if pretrained is None:
            pretrained = hp.Choice('pretrained',
                                   [True, False],
                                   default=False)
        is_embedding_trainable = self.is_embedding_trainable
        if is_embedding_trainable is None:
            is_embedding_trainable = hp.Choice('is_embedding_trainable',
                                               [True, False],
                                               default=False)
        embedding_dim = hp.Choice('embedding_dim',
                                  [32, 64, 128, 256, 512],
                                  default=128)
        if pretrained:
            # TODO: load from pretrained weights
            layer = tf.keras.layers.Embedding(
                input_dim=input_node.shape[1],
                output_dim=embedding_dim,
                input_length=const.Constant.VOCABULARY_SIZE,
                trainable=is_embedding_trainable)
            # weights=[embedding_matrix])
        else:
            layer = tf.keras.layers.Embedding(
                input_dim=input_node.shape[1],
                output_dim=embedding_dim,
                input_length=const.Constant.VOCABULARY_SIZE,
                trainable=is_embedding_trainable)
        return layer(input_node)


class TextBlock(RNNBlock):
    pass


class StructuredBlock(HyperBlock):

    def build(self, hp, inputs=None):
        pass


class TimeSeriesBlock(HyperBlock):

    def build(self, hp, inputs=None):
        pass


class GeneralBlock(HyperBlock):

    def build(self, hp, inputs=None):
        pass
