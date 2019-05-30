from abc import ABC

import tensorflow as tf
from autokeras.hypermodel.hyper_block import HyperBlock
from autokeras.layer_utils import flatten


class HyperHead(HyperBlock, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = None


class ClassificationHead(HyperHead):
    def build(self, hp, inputs=None):
        input_node = self._format_inputs(inputs, 1)[0]
        output_node = input_node
        output_node = flatten(output_node)

        output_node = tf.keras.layers.Dense(self.output_shape)(output_node)
        output_node = tf.keras.layers.Softmax()(output_node)

        # TODO: Add hp.Choice to use sigmoid

        return tf.keras.Model(input_node, output_node)


class RegressionHead(HyperHead):
    def build(self, hp, inputs=None):
        input_node = self._format_inputs(inputs, 1)[0]
        output_node = input_node
        output_node = flatten(output_node)
        output_node = tf.keras.layers.Dense(self.output_shape)(output_node)

        return tf.keras.Model(input_node, output_node)


class TensorRegressionHead(HyperHead):
    def build(self, hp, inputs=None):
        pass


class TensorClassificationHead(HyperHead):
    def build(self, hp, inputs=None):
        pass
