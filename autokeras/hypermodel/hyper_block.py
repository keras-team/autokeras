from abc import ABC

import tensorflow as tf

from autokeras import HyperModel
from autokeras.hypermodel.hyper_node import HyperNode
from autokeras.hyperparameters import HyperParameters
from autokeras.layer_utils import flatten, format_inputs


class HierarchicalHyperParameters(HyperParameters):
    def retrieve(self, name, type, config):
        return super().retrieve(tf.get_default_graph().get_name_scope() + '/' + name, type, config)


class HyperBlock(HyperModel, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.name:
            prefix = self.__class__.__name__
            self.name = prefix + '_' + str(tf.keras.backend.get_uid(prefix))
        self.inputs = None
        self.outputs = None
        self.n_output_node = 1

    def __call__(self, inputs):
        self.inputs = format_inputs(inputs, self.name)
        for input_node in self.inputs:
            input_node.add_out_hypermodel(self)
        self.outputs = []
        for _ in range(self.n_output_node):
            output_node = HyperNode()
            output_node.add_in_hypermodel(self)
            self.outputs.append(output_node)
        return self.outputs

    def build_output(self, hp, inputs=None):
        raise NotImplementedError

    def build(self, hp, inputs=None, sub_model=False):
        if sub_model:
            with tf.name_scope(self.name):
                outputs = self.build_output(hp, inputs)
            return outputs
        outputs = self.build_output(hp, inputs)
        return tf.keras.Model(inputs, outputs)


class ResNetBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass


class DenseNetBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass


class MlpBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        input_node = format_inputs(inputs, 1)[0]
        output_node = input_node
        output_node = flatten(output_node)

        for i in range(hp.Choice('num_layers', [1, 2, 3], default=2)):
            output_node = tf.keras.layers.Dense(hp.Choice('units_{i}'.format(i=i),
                                                          [16, 32, 64],
                                                          default=32))(output_node)

        return output_node


class AlexNetBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass


class CnnBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass


class RnnBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass


class LstmBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass


class SeqToSeqBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass


class ImageBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass


class NlpBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass


class Merge(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass


class XceptionBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass
