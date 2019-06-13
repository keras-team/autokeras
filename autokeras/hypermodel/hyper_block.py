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
        active_category = hp.Choice('activate_category', ['softmax','relu','tanh','sigmoid'], default='relu')
        layer_stack = hp.Choice('layer_stack', ['dense-bn-act', 'dense-act', 'act-bn-dense'],default='act-bn-dense')
        for i in range(hp.Choice('num_layers', [1, 2, 3], default=2)):
            if layer_stack == 'dense-bn-act':
                output_node = tf.keras.layers.Dense(hp.Choice('units_{i}'.format(i=i),
                                                              [16, 32, 64, 128, 256, 512, 1024],
                                                              default=32))(output_node)
                output_node = tf.keras.layers.BatchNormalization()(output_node)
                output_node = tf.keras.layers.Activation(active_category)(output_node)
                output_node = tf.keras.layers.Dropout(rate=hp.Choice('dropout_rate', [0, 0.25, 0.5], default=0.5))(output_node)
            elif layer_stack == 'dense-act':
                output_node = tf.keras.layers.Dense(hp.Choice('units_{i}'.format(i=i),
                                                              [16, 32, 64, 128, 256, 512, 1024],
                                                              default=32))(output_node)
                output_node = tf.keras.layers.Activation(active_category)(output_node)
                output_node = tf.keras.layers.Dropout(rate=hp.Choice('dropout_rate', [0, 0.25, 0.5], default=0.5))(output_node)
            else:
                output_node = tf.keras.layers.Activation(active_category)(output_node)
                output_node = tf.keras.layers.BatchNormalization()(output_node)
                output_node = tf.keras.layers.Dense(hp.Choice('units_{i}'.format(i=i),
                                                              [16, 32, 64, 128, 256, 512, 1024],
                                                              default=32))(output_node)
                output_node = tf.keras.layers.Dropout(rate=hp.Choice('dropout_rate', [0, 0.25, 0.5], default=0.5))(output_node)
        return output_node


class RNNBlock(HyperBlock):

    def build(self, hp, inputs=None):
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        layer_units = input_node.shape[1]
        output_node = tf.reshape(Flatten().build(hp, output_node), [-1, layer_units, 1])
        type_of_block = hp.Choice('rnn_type', ['vanilla', 'gru', 'lstm'], default='vanilla')
        if type_of_block == 'vanilla':
            print("make vanilla")
            in_layer = tf.keras.layers.SimpleRNN
        elif type_of_block == 'gru':
            print("make gru")
            in_layer = tf.keras.layers.GRU
        elif type_of_block == 'lstm':
            print("make lstm")
            in_layer = tf.keras.layers.LSTM

        choice_of_layers = hp.Choice('num_layers', [1, 2, 3], default=2)
        for i in range(choice_of_layers):
            if i == choice_of_layers - 1:
                bidirectional_block = tf.keras.layers.Bidirectional(
                    in_layer(layer_units, activation='tanh', use_bias=True,
                                              return_sequences=False))
            else:
                bidirectional_block = tf.keras.layers.Bidirectional(
                    in_layer(layer_units, activation='tanh', use_bias=True,
                                              return_sequences=True))
            output_node = bidirectional_block(output_node)

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
        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        # To decide the size of the architecture
        filter_entry = hp.Choice('num_filters', [16, 32, 64], default=32)
        filter_middle = int(filter_entry * 22.75)
        filter_exit = int(filter_entry * 48)

        # Entry flow (original filters: 32, 64)
        output_node = tf.keras.layers.Conv2D(filter_entry, (3, 3), strides=(2, 2), use_bias=False)(output_node)
        output_node = tf.keras.layers.BatchNormalization(axis=channel_axis)(output_node)
        output_node = tf.keras.layers.Activation('relu')(output_node)
        output_node = tf.keras.layers.Conv2D(filter_entry*2, (3, 3), use_bias=False)(output_node)
        output_node = tf.keras.layers.BatchNormalization(axis=channel_axis)(output_node)
        output_node = tf.keras.layers.Activation('relu')(output_node)

        # Entry flow (original filters: 128, 256, 728)
        output_node = self._entry_flow(filter_entry*4, channel_axis, start_with_relu=False)(output_node)
        output_node = self._entry_flow(filter_entry*8, channel_axis, start_with_relu=True)(output_node)
        output_node = self._entry_flow(filter_middle, channel_axis, start_with_relu=True)(output_node)

        # Middle flow (original filters: 728), repeated eight times
        for i in range(8):
            output_node = self._middle_flow(filter_middle, channel_axis)(output_node)

        # Exit flow (original filters: 728, 1024; 1536, 2048)
        output_node = self._exit_flow(filter_middle, filter_entry*32,
                                      start_with_relu=True, with_shortcut=True)(output_node)
        output_node = self._exit_flow(filter_exit, filter_entry*64,
                                      start_with_relu=False, with_shortcut=False)(output_node)

        return output_node

    def _entry_flow(self, filters, channel_axis=-1, start_with_relu=True):
        def func(x):
            residual = tf.keras.layers.Conv2D(filters, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
            residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)
            if start_with_relu:
                x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
            x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
            return tf.keras.layers.add([x, residual])
        return func

    def _middle_flow(self, filters, channel_axis=-1):
        def func(x):
            residual = x
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
            return tf.keras.layers.add([x, residual])
        return func

    def _exit_flow(self, filter1, filter2, channel_axis=-1, start_with_relu=False, with_shortcut=False):
        def func(x):
            if with_shortcut:
                residual = tf.keras.layers.Conv2D(filter2, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
                residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

            if start_with_relu:
                x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filter1, (3, 3), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filter2, (3, 3), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
            if not start_with_relu:
                x = tf.keras.layers.Activation('relu')(x)

            if with_shortcut:
                x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
                x = tf.keras.layers.add([x, residual])
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
