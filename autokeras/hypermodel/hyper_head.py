from abc import ABC

import tensorflow as tf
from autokeras.hypermodel.hyper_block import HyperBlock, Flatten
from autokeras.layer_utils import format_inputs


class HyperHead(HyperBlock, ABC):
    def __init__(self, output_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape


class ClassificationHead(HyperHead):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, hp, inputs=None):
        input_node = format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        output_node = Flatten().build(hp, output_node)

        output_node = tf.keras.layers.Dense(self.output_shape)(output_node)
        output_node = tf.keras.layers.Softmax()(output_node)

        # TODO: Add hp.Choice to use sigmoid

        return tf.keras.Model(input_node, output_node)


class RegressionHead(HyperHead):
    def build(self, hp, inputs=None):
        input_node = format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        output_node = Flatten().build(hp, output_node)
        output_node = tf.keras.layers.Dense(self.output_shape[-1])(output_node)

        return output_node
