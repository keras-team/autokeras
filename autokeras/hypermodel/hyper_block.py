from abc import ABC

import tensorflow as tf

from autokeras import HyperModel
from autokeras.hypermodel.hyper_node import HyperNode
from autokeras.hyperparameters import HyperParameters
from autokeras.layer_utils import format_inputs, get_global_average_pooling_layer_class


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
        input_node = format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        output_node = Flatten().build(hp, output_node, sub_model=True)

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


def shape_compatible(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    # TODO: If they can be the same after passing through any layer, they are compatible.
    #  e.g. (32, 32, 3), (16, 16, 2) are compatible
    return shape1[:-1] == shape2[:-1]


class Merge(HyperBlock):
    def build_output(self, hp, inputs=None):
        inputs = format_inputs(inputs, self.name)
        if len(inputs) == 1:
            return inputs

        if not all([shape_compatible(input_node.shape, inputs[0].shape) for input_node in inputs]):
            new_inputs = []
            for input_node in inputs:
                new_inputs.append(Flatten().build(hp, input_node, sub_model=True))
            inputs = new_inputs

        # TODO: Even inputs have different shape[-1], they can still be Add() after another layer.
        # Check if the inputs are all of the same shape
        if all([input_node.shape == inputs[0].shape for input_node in inputs]):
            if hp.Choice("merge_type", ['Add', 'Concatenate'], default='Add'):
                return tf.keras.layers.Add(inputs)

        return tf.keras.layers.Add()(inputs)


class XceptionBlock(HyperBlock):
    def build_output(self, hp, inputs=None):
        pass


class Flatten(HyperBlock):
    def build_output(self, hp, inputs=None):
        input_node = format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        if len(output_node.shape) > 5:
            raise ValueError("Expect the input tensor to have less or equal to 5 dimensions, "
                             "but got {shape}".format(shape=output_node.shape))
        # Flatten the input tensor
        # TODO: Add hp.Choice to use Flatten()
        if len(output_node.shape) > 2:
            global_average_pooling = get_global_average_pooling_layer_class(output_node.shape)
            output_node = global_average_pooling()(output_node)
        return output_node
