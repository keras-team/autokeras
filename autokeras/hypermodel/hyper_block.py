from abc import ABC

import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import HyperModel
from autokeras.hypermodel.hyper_node import HyperNode
from autokeras.hyperparameters import HyperParameters
from autokeras.layer_utils import flatten


class HierarchicalHyperParameters(HyperParameters):
    def retrieve(self, name, type, config):
        super().retrieve(tf.get_default_graph().get_name_scope() + '/' + name, type, config)


class HyperBlock(HyperModel, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.name:
            prefix = self.__class__.__name__
            self.name = prefix + '_' + str(tf.keras.backend.get_uid(prefix))
        self.inputs = None
        self.outputs = None
        self.n_output_node = 1
        self.build = self._build_with_name_scope

    def __call__(self, inputs):
        self.inputs = inputs
        for input_node in inputs:
            input_node.add_out_hypermodel(self)
        self.outputs = []
        for _ in range(self.n_output_node):
            output_node = HyperNode()
            output_node.add_in_hypermodel(self)
            self.outputs.append(output_node)
        return self.outputs

    def build(self, hp, inputs=None):
        raise NotImplementedError

    def _build_with_name_scope(self, hp, inputs=None):
        with tf.name_scope(self.name):
            self.build(hp, inputs)

    def _format_inputs(self, inputs, num):
        inputs = nest.flatten(inputs)

        if isinstance(inputs, list) and len(inputs) == num:
            return inputs

        if (not isinstance(inputs, list)) and num == 1:
            return inputs

        raise ValueError('Expected {num} input in the '
                         'inputs list for hypermodel {name} '
                         'but received {len} inputs.'.format(num=num, name=self.name, len=len(inputs)))


class ResNetBlock(HyperBlock):
    def build(self, hp, inputs=None):
        pass


class DenseNetBlock(HyperBlock):
    def build(self, hp, inputs=None):
        pass


class MlpBlock(HyperBlock):
    def build(self, hp, inputs=None):
        input_node = self._format_inputs(inputs, 1)[0]
        output_node = input_node
        output_node = flatten(output_node)

        for i in range(hp.Choice('num_layers', [1, 2, 3], default=2)):
            output_node = tf.keras.layers.Dense(hp.Choice('units_{i}'.format(i=i)))(output_node)

        return tf.keras.Model(input_node, output_node)


class AlexNetBlock(HyperBlock):
    def build(self, hp, inputs=None):
        pass


class CnnBlock(HyperBlock):
    def build(self, hp, inputs=None):
        pass


class RnnBlock(HyperBlock):
    def build(self, hp, inputs=None):
        pass


class LstmBlock(HyperBlock):
    def build(self, hp, inputs=None):
        pass


class SeqToSeqBlock(HyperBlock):
    def build(self, hp, inputs=None):
        pass


class ImageBlock(HyperBlock):
    def build(self, hp, inputs=None):
        pass


class NlpBlock(HyperBlock):
    def build(self, hp, inputs=None):
        pass


class Merge(HyperBlock):
    def build(self, hp, inputs=None):
        pass


class XceptionBlock(HyperBlock):
    def build(self, hp, inputs=None):
        pass
