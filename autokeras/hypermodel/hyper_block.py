import tensorflow as tf

from autokeras.hypermodel import hyper_node, hypermodel
from autokeras import layer_utils


class HyperBlock(hypermodel.HyperModel):

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._conv = None
        self._bottleneck = None

    def build(self, hp, inputs=None):
        input_node = inputs[0]
        self._conv = layer_utils.get_conv_class(input_node.shape)
        planes = hp.Choice('first_layer_width', [16, 32, 64, 128, 256, 512, 1024], default=16)
        self._bottleneck = hp.Choice('bottleneck', [True, False], default=True)

        output_node = input_node
        output_node = self._conv(planes, padding='same', kernel_size=3)(output_node)
        output_node = tf.keras.layers.BatchNormalization()(output_node)
        output_node = tf.keras.layers.ReLU()(output_node)

        output_node = self._make_layer(planes, hp.Choice('layer_1', [2, 3], default=2), output_node, 1)
        planes *= 2
        output_node = self._make_layer(planes, hp.Choice('layer_2', [2, 4, 8], default=2), output_node, 2)
        planes *= 2
        output_node = self._make_layer(planes, hp.Choice('layer_3', [2, 6, 23, 36], default=2), output_node, 2)
        planes *= 2
        output_node = self._make_layer(planes, hp.Choice('layer_4', [2, 3], default=2), output_node, 2)

        return output_node

    def _make_basic_block(self, planes, input_node, stride=1):
        output_node = input_node
        output_node = self._conv(planes, kernel_size=3, padding='same', strides=stride)(output_node)
        output_node = tf.keras.layers.BatchNormalization()(output_node)
        output_node = tf.keras.layers.ReLU()(output_node)
        output_node = self._conv(planes, kernel_size=3, padding='same')(output_node)
        output_node = tf.keras.layers.BatchNormalization()(output_node)
        return output_node

    def _make_bottleneck_block(self, planes, input_node, stride=1):
        output_node = input_node
        output_node = self._conv(planes, kernel_size=1, padding='same')(output_node)
        output_node = tf.keras.layers.BatchNormalization(planes)(output_node)
        output_node = tf.keras.layers.ReLU()(output_node)
        output_node = self._conv(planes, kernel_size=3, padding='same', strides=stride)(output_node)
        output_node = tf.keras.layers.BatchNormalization()(output_node)
        output_node = tf.keras.layers.ReLU()(output_node)
        output_node = self._conv(4 * planes, kernel_size=1, padding='same')(output_node)
        output_node = tf.keras.layers.BatchNormalization()(output_node)
        return output_node

    def _make_layer(self, planes, blocks, input_node, stride):
        strides = [stride] + [1] * (blocks - 1)
        output_node = input_node
        for current_stride in strides:
            residual_node = output_node
            if self._bottleneck:
                output_node = self._make_basic_block(planes, output_node, current_stride)
            else:
                output_node = self._make_bottleneck_block(planes, output_node, current_stride)
            if current_stride != 1 or residual_node.shape[-1] != output_node.shape[-1]:
                residual_node = self._conv(output_node.shape[-1],
                                           kernel_size=1,
                                           padding='same',
                                           strides=current_stride)(residual_node)
                residual_node = tf.keras.layers.BatchNormalization()(residual_node)

            output_node = tf.keras.layers.Add()([output_node, residual_node])
            output_node = tf.keras.layers.ReLU()(output_node)
        return output_node


class DenseBlock(HyperBlock):

    def build(self, hp, inputs=None):
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        output_node = Flatten().build(hp, output_node)

        for i in range(hp.Choice('num_layers', [1, 2, 3], default=2)):
            output_node = tf.keras.layers.Dense(hp.Choice('units_{i}'.format(i=i),
                                                          [16, 32, 64],
                                                          default=32))(output_node)
        return output_node


class RNNBlock(HyperBlock):

    def build(self, hp, inputs=None):
        pass


class ImageBlock(HyperBlock):

    def build(self, hp, inputs=None):
        # TODO: make it more advanced, selecting from multiple models, e.g., ResNet.
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node

        for i in range(hp.Choice('num_layers', [1, 2, 3], default=2)):
            output_node = tf.keras.layers.Conv2D(hp.Choice('units_{i}'.format(i=i),
                                                           [16, 32, 64],
                                                           default=32),
                                                 hp.Choice('kernel_size_{i}'.format(i=i),
                                                           [3, 5, 7],
                                                           default=3))(output_node)
        return output_node


def shape_compatible(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    # TODO: If they can be the same after passing through any layer, they are compatible.
    #  e.g. (32, 32, 3), (16, 16, 2) are compatible
    return shape1[:-1] == shape2[:-1]


class Merge(HyperBlock):

    def build(self, hp, inputs=None):
        inputs = layer_utils.format_inputs(inputs, self.name)
        if len(inputs) == 1:
            return inputs

        if not all([shape_compatible(input_node.shape, inputs[0].shape) for input_node in inputs]):
            new_inputs = []
            for input_node in inputs:
                new_inputs.append(Flatten().build(hp, input_node))
            inputs = new_inputs

        # TODO: Even inputs have different shape[-1], they can still be Add() after another layer.
        # Check if the inputs are all of the same shape
        if all([input_node.shape == inputs[0].shape for input_node in inputs]):
            if hp.Choice("merge_type", ['Add', 'Concatenate'], default='Add'):
                return tf.keras.layers.Add(inputs)

        return tf.keras.layers.Add()(inputs)


class XceptionBlock(HyperBlock):

    def build(self, hp, inputs=None):
        pass


class Flatten(HyperBlock):

    def build(self, hp, inputs=None):
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        if len(output_node.shape) > 5:
            raise ValueError("Expect the input tensor to have less or equal to 5 dimensions, "
                             "but got {shape}".format(shape=output_node.shape))
        # Flatten the input tensor
        # TODO: Add hp.Choice to use Flatten()
        if len(output_node.shape) > 2:
            global_average_pooling = layer_utils.get_global_average_pooling_class(output_node.shape)
            output_node = global_average_pooling()(output_node)
        return output_node


class Reshape(HyperBlock):

    def __init__(self, output_shape, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape

    def build(self, hp, inputs=None):
        # TODO: Implement reshape layer
        return inputs
