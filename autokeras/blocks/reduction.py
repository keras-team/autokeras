from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.util import nest

from autokeras.engine import block as block_module
from autokeras.utils import layer_utils
from autokeras.utils import utils


def shape_compatible(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    # TODO: If they can be the same after passing through any layer,
    #  they are compatible. e.g. (32, 32, 3), (16, 16, 2) are compatible
    return shape1[:-1] == shape2[:-1]


class Merge(block_module.Block):
    """Merge block to merge multiple nodes into one.

    # Arguments
        merge_type: String. 'add' or 'concatenate'. If left unspecified, it will be
            tuned automatically.
    """

    def __init__(self, merge_type: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.merge_type = merge_type

    def get_config(self):
        config = super().get_config()
        config.update({'merge_type': self.merge_type})
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        if len(inputs) == 1:
            return inputs

        merge_type = self.merge_type or hp.Choice('merge_type',
                                                  ['add', 'concatenate'],
                                                  default='add')

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
            if merge_type == 'add':
                return layers.Add(inputs)

        return layers.Concatenate()(inputs)


class Flatten(block_module.Block):
    """Flatten the input tensor with Keras Flatten layer."""

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        if len(input_node.shape) > 2:
            return layers.Flatten()(input_node)
        return input_node


class SpatialReduction(block_module.Block):
    """Reduce the dimension of a spatial tensor, e.g. image, to a vector.

    # Arguments
        reduction_type: String. 'flatten', 'global_max' or 'global_avg'.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, reduction_type: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.reduction_type = reduction_type

    def get_config(self):
        config = super().get_config()
        config.update({'reduction_type': self.reduction_type})
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        # No need to reduce.
        if len(output_node.shape) <= 2:
            return output_node

        reduction_type = self.reduction_type or hp.Choice('reduction_type',
                                                          ['flatten',
                                                           'global_max',
                                                           'global_avg'],
                                                          default='global_avg')
        if reduction_type == 'flatten':
            output_node = Flatten().build(hp, output_node)
        elif reduction_type == 'global_max':
            output_node = layer_utils.get_global_max_pooling(
                output_node.shape)()(output_node)
        elif reduction_type == 'global_avg':
            output_node = layer_utils.get_global_average_pooling(
                output_node.shape)()(output_node)
        return output_node


class TemporalReduction(block_module.Block):
    """Reduce the dimension of a temporal tensor, e.g. output of RNN, to a vector.

    # Arguments
        reduction_type: String. 'flatten', 'global_max' or 'global_avg'. If left
            unspecified, it will be tuned automatically.
    """

    def __init__(self, reduction_type: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.reduction_type = reduction_type

    def get_config(self):
        config = super().get_config()
        config.update({'reduction_type': self.reduction_type})
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        # No need to reduce.
        if len(output_node.shape) <= 2:
            return output_node

        reduction_type = self.reduction_type or hp.Choice('reduction_type',
                                                          ['flatten',
                                                           'global_max',
                                                           'global_avg'],
                                                          default='global_avg')

        if reduction_type == 'flatten':
            output_node = Flatten().build(hp, output_node)
        elif reduction_type == 'global_max':
            output_node = tf.math.reduce_max(output_node, axis=-2)
        elif reduction_type == 'global_avg':
            output_node = tf.math.reduce_mean(output_node, axis=-2)
        elif reduction_type == 'global_min':
            output_node = tf.math.reduce_min(output_node, axis=-2)

        return output_node
