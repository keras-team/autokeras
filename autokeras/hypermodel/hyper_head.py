from abc import ABC

import tensorflow as tf
from autokeras.hypermodel.hyper_block import HyperBlock
from autokeras.layer_utils import flatten, format_inputs


class HyperHead(HyperBlock, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = None


class ClassificationHead(HyperHead):
    def build_output(self, hp, inputs=None):
        input_node = format_inputs(inputs, 1)[0]
        output_node = input_node
        output_node = flatten(output_node)

        output_node = tf.keras.layers.Dense(self.output_shape)(output_node)
        output_node = tf.keras.layers.Softmax()(output_node)

        # TODO: Add hp.Choice to use sigmoid

        return tf.keras.Model(input_node, output_node)


class RegressionHead(HyperHead):
    def build_output(self, hp, inputs=None):
        input_node = format_inputs(inputs, 1)[0]
        output_node = input_node
        output_node = flatten(output_node)
        output_node = tf.keras.layers.Dense(self.output_shape[-1])(output_node)

        return output_node


class TensorRegressionHead(HyperHead):
    def build_output(self, hp, inputs=None):
        pass


class TensorClassificationHead(HyperHead):
    def build_output(self, hp, inputs=None):
        pass
