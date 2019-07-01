import numpy as np
import tensorflow as tf
import kerastuner

from autokeras.hypermodel import hyper_node
from autokeras import layer_utils


class HyperBlock(kerastuner.HyperModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = None
        self.outputs = None
        self._num_output_node = 1

    def __call__(self, inputs):
        self.inputs = layer_utils.format_inputs(inputs, self.name)
        for input_node in self.inputs:
            input_node.add_out_hypermodel(self)
        self.outputs = []
        for _ in range(self._num_output_node):
            output_node = hyper_node.Node()
            output_node.add_in_hypermodel(self)
            self.outputs.append(output_node)
        return self.outputs

    def build(self, hp, inputs=None):
        raise NotImplementedError


class ResNetBlock(HyperBlock):

    def build(self, hp, inputs=None):
        pass


class DenseBlock(HyperBlock):

    def build(self, hp, inputs=None):
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        output_node = Flatten().build(hp, output_node)
        active_category = hp.Choice(
            'activate_category',
            ['softmax', 'relu', 'tanh', 'sigmoid'],
            default='relu')
        layer_stack = hp.Choice(
            'layer_stack',
            ['dense-bn-act', 'dense-act', 'act-bn-dense'],
            default='act-bn-dense')
        for i in range(hp.Choice('num_layers', [1, 2, 3], default=2)):
            if layer_stack == 'dense-bn-act':
                output_node = tf.keras.layers.Dense(hp.Choice(
                    'units_{i}'.format(i=i),
                    [16, 32, 64, 128, 256, 512, 1024],
                    default=32))(output_node)
                output_node = tf.keras.layers.BatchNormalization()(output_node)
                output_node = tf.keras.layers.Activation(active_category)(
                    output_node)
                output_node = tf.keras.layers.Dropout(
                    rate=hp.Choice('dropout_rate', [0, 0.25, 0.5], default=0.5))(
                    output_node)
            elif layer_stack == 'dense-act':
                output_node = tf.keras.layers.Dense(
                    hp.Choice('units_{i}'.format(i=i),
                              [16, 32, 64, 128, 256, 512, 1024],
                              default=32))(output_node)
                output_node = tf.keras.layers.Activation(active_category)(
                    output_node)
                output_node = tf.keras.layers.Dropout(
                    rate=hp.Choice('dropout_rate', [0, 0.25, 0.5], default=0.5))(
                    output_node)
            else:
                output_node = tf.keras.layers.Activation(active_category)(
                    output_node)
                output_node = tf.keras.layers.BatchNormalization()(output_node)
                output_node = tf.keras.layers.Dense(
                    hp.Choice('units_{i}'.format(i=i),
                              [16, 32, 64, 128, 256, 512, 1024],
                              default=32))(output_node)
                output_node = tf.keras.layers.Dropout(
                    rate=hp.Choice('dropout_rate', [0, 0.25, 0.5], default=0.5))(
                    output_node)
        return output_node


class RNNBlock(HyperBlock):

    def __init__(self, bidirectional=None, return_sequences=False, **kwargs):
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences

        super().__init__(**kwargs)

    def build(self, hp, inputs=None):
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        input_node = layer_utils.validate_input(input_node)
        feature_size = input_node.shape[2]

        if self.bidirectional is None:
            self.bidirectional = hp.Choice('bidirectional', [True, False], default=False)
        bidirectional_layer = tf.keras.layers.Bidirectional if self.bidirectional else tf.keras.layers.Lambda(
            lambda x: x)

        # TODO: Attention layer can also be placed after LSTM.
        #       Possible values for hp.Choice must be [attention_first, attention_last, no_attention]
        attention_mode = hp.Choice('attention', [True, False], default=True)
        input_node = layer_utils.attention_block(input_node) if attention_mode else input_node

        # attention not enabled on Vanilla rnn
        if attention_mode:
            rnn_type = hp.Choice('rnn_type', ['gru', 'lstm'], default='lstm')
        else:
            rnn_type = hp.Choice('rnn_type', ['vanilla', 'gru', 'lstm'], default='lstm')
        in_block = layer_utils.get_rnn_block(rnn_type)
        choice_of_layers = hp.Choice('num_layers', [1, 2, 3], default=2)

        print("HP choices here \n Bid : ", self.bidirectional, " attention : ", attention_mode, " num_layers : ",
              choice_of_layers, " rnn_type : ", rnn_type)

        output_node = input_node
        for i in range(choice_of_layers):
            return_sequences = self.return_sequences if i == choice_of_layers - 1 else True
            block = bidirectional_layer(in_block(feature_size, activation='tanh', return_sequences=return_sequences))
            output_node = block(output_node)

        output_node = Flatten().build(hp, output_node)

        # return_sequences does not necessarily need to be True
        # for attention to work; the underlying computation is the same, and return_sequences should be used only based
        # on whether you need 1 output or an output "for each timestep".
        return output_node


class S2SBlock(HyperBlock):

    def __init__(self, type="auto_enc", **kwargs):
        if type not in layer_utils.get_s2s_types():
            raise ValueError("Invalid type. Allowed types are : ", layer_utils.get_s2s_types(), " \n but got ",
                             type)
        self.seq_type = type
        super().__init__(**kwargs)

    def build(self, hp, inputs=None, targets=None):
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        input_node = layer_utils.validate_input(input_node)
        feature_size = input_node.shape[-1]
        if self.seq_type == 'auto_enc':
            target_node = input_node
        else:
            if targets is None:
                raise ValueError("Model needs valid target sequences for training. No targets passed")
            else:
                target_node = layer_utils.format_inputs(targets, self.name, num=1)[0]

        # TODO: Attention layer can also be placed after LSTM.
        #       Possible values for hp.Choice must be [attention_first, attention_last, no_attention]
        attention_mode = hp.Choice('attention', [True, False], default=True)
        input_node = layer_utils.attention_block(input_node) if attention_mode else input_node

        rnn_type = hp.Choice('rnn_type', ['gru', 'lstm'], default='lstm')
        choice_of_layers = hp.Choice('num_layers', [1, 2, 3], default=2)

        output_node = layer_utils.seq2seq_builder(input_node,target_node,rnn_type,choice_of_layers,feature_size)

        return output_node


class ImageBlock(HyperBlock):

    def build(self, hp, inputs=None):
        # TODO: make it more advanced, selecting from multiple models, e.g.,
        #  ResNet.
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node

        for i in range(hp.Choice('num_layers', [1, 2, 3], default=2)):
            output_node = tf.keras.layers.Conv2D(
                hp.Choice('units_{i}'.format(i=i),
                          [16, 32, 64],
                          default=32),
                hp.Choice('kernel_size_{i}'.format(i=i),
                          [3, 5, 7],
                          default=3))(output_node)
        return output_node


def shape_compatible(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    # TODO: If they can be the same after passing through any layer,
    #  they are compatible. e.g. (32, 32, 3), (16, 16, 2) are compatible
    return shape1[:-1] == shape2[:-1]


class Merge(HyperBlock):

    def build(self, hp, inputs=None):
        inputs = layer_utils.format_inputs(inputs, self.name)
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


class XceptionBlock(HyperBlock):
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

    def build(self, hp, inputs=None):
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        channel_axis = 1 \
            if tf.keras.backend.image_data_format() == 'channels_first' else -1

        # Parameters
        # [general]
        kernel_size = hp.Range("kernel_size", 3, 5)
        initial_strides = (2, 2)
        activation = hp.Choice("activation", ["relu", "selu"])
        # [Entry Flow]
        conv2d_filters = hp.Choice("conv2d_num_filters", [32, 64, 128])
        sep_filters = hp.Range("sep_num_filters", 128, 768)
        # [Middle Flow]
        residual_blocks = hp.Range("num_residual_blocks", 2, 8)
        # [Exit Flow]

        # Model
        # Initial conv2d
        dims = conv2d_filters
        output_node = self._conv(dims, kernel_size=(kernel_size, kernel_size),
                                 activation=activation, strides=initial_strides)(
            output_node)
        # Separable convs
        dims = sep_filters
        for _ in range(residual_blocks):
            output_node = self._residual(dims, activation=activation,
                                         max_pooling=False,
                                         channel_axis=channel_axis)(output_node)
        # Exit
        dims *= 2
        output_node = self._residual(dims, activation=activation,
                                     max_pooling=True, channel_axis=channel_axis)(
            output_node)

        return output_node

    @classmethod
    def _sep_conv(cls, filters, kernel_size=(3, 3), activation='relu'):
        def func(x):
            if activation == 'selu':
                x = tf.keras.layers.SeparableConv2D(
                    filters, kernel_size,
                    activation='selu',
                    padding='same',
                    kernel_initializer='lecun_normal')(x)
            elif activation == 'relu':
                x = tf.keras.layers.SeparableConv2D(filters, kernel_size,
                                                    padding='same',
                                                    use_bias=False)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation('relu')(x)
            else:
                raise ValueError(
                    'Unknown activation function: {:s}'.format(activation))
            return x

        return func

    @classmethod
    def _residual(cls,
                  filters, kernel_size=(3, 3), activation='relu',
                  pool_strides=(2, 2), max_pooling=True,
                  channel_axis=-1):
        """ Residual block. """

        def func(x):
            if max_pooling:
                res = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1),
                                             strides=pool_strides, padding='same')(x)
            elif filters != tf.keras.backend.int_shape(x)[channel_axis]:
                res = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1),
                                             padding='same')(x)
            else:
                res = x

            x = cls._sep_conv(filters, kernel_size, activation)(x)
            x = cls._sep_conv(filters, kernel_size, activation)(x)
            if max_pooling:
                x = tf.keras.layers.MaxPool2D(kernel_size, strides=pool_strides,
                                              padding='same')(x)

            x = tf.keras.layers.add([x, res])
            return x

        return func

    @classmethod
    def _conv(cls, filters, kernel_size=(3, 3), activation='relu', strides=(2, 2)):
        """ 2d convolution block. """

        def func(x):
            if activation == 'selu':
                x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                           activation='selu',
                                           padding='same',
                                           kernel_initializer='lecun_normal',
                                           bias_initializer='zeros')(x)
            elif activation == 'relu':
                x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                           padding='same', use_bias=False)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation('relu')(x)
            else:
                raise ValueError(
                    'Unknown activation function: {:s}'.format(activation))
            return x

        return func

    @classmethod
    def _dense(cls, dims, activation='relu', batchnorm=True, dropout_rate=0):
        def func(x):
            if activation == 'selu':
                x = tf.keras.layers.Dense(dims, activation='selu',
                                          kernel_initializer='lecun_normal',
                                          bias_initializer='zeros')(x)
                if dropout_rate:
                    x = tf.keras.layers.AlphaDropout(dropout_rate)(x)
            elif activation == 'relu':
                x = tf.keras.layers.Dense(dims, activation='relu')(x)
                if batchnorm:
                    x = tf.keras.layers.BatchNormalization()(x)
                if dropout_rate:
                    x = tf.keras.layers.Dropout(dropout_rate)(x)
            else:
                raise ValueError(
                    'Unknown activation function: {:s}'.format(activation))
            return x

        return func


class Flatten(HyperBlock):

    def build(self, hp, inputs=None):
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        if len(output_node.shape) > 5:
            raise ValueError(
                'Expect the input tensor to have less or equal to 5 '
                'dimensions, but got {shape}'.format(shape=output_node.shape))
        # Flatten the input tensor
        # TODO: Add hp.Choice to use Flatten()
        if len(output_node.shape) > 2:
            global_average_pooling = \
                layer_utils.get_global_average_pooling_layer_class(
                    output_node.shape)
            output_node = global_average_pooling()(output_node)
        return output_node


class Reshape(HyperBlock):

    def __init__(self, output_shape, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape

    def build(self, hp, inputs=None):
        # TODO: Implement reshape layer
        return inputs
