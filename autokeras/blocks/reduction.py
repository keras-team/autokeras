# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.util import nest

from autokeras.engine import block as block_module
from autokeras.utils import layer_utils
from autokeras.utils import utils

REDUCTION_TYPE = "reduction_type"
FLATTEN = "flatten"
GLOBAL_MAX = "global_max"
GLOBAL_AVG = "global_avg"


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
        config.update({"merge_type": self.merge_type})
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        if len(inputs) == 1:
            return inputs

        if not all(
            [
                shape_compatible(input_node.shape, inputs[0].shape)
                for input_node in inputs
            ]
        ):
            new_inputs = []
            for input_node in inputs:
                new_inputs.append(Flatten().build(hp, input_node))
            inputs = new_inputs

        # TODO: Even inputs have different shape[-1], they can still be Add(
        #  ) after another layer. Check if the inputs are all of the same
        #  shape
        if self._inputs_same_shape(inputs):
            merge_type = self.merge_type or hp.Choice(
                "merge_type", ["add", "concatenate"], default="add"
            )
            if merge_type == "add":
                return layers.Add()(inputs)

        return layers.Concatenate()(inputs)

    def _inputs_same_shape(self, inputs):
        for input_node in inputs:
            if input_node.shape.as_list() != inputs[0].shape.as_list():
                return False
        return True


class Flatten(block_module.Block):
    """Flatten the input tensor with Keras Flatten layer."""

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        if len(input_node.shape) > 2:
            return layers.Flatten()(input_node)
        return input_node


class Reduction(block_module.Block):
    def __init__(self, reduction_type: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.reduction_type = reduction_type

    def get_config(self):
        config = super().get_config()
        config.update({REDUCTION_TYPE: self.reduction_type})
        return config

    def global_max(self, input_node):
        raise NotImplementedError

    def global_avg(self, input_node):
        raise NotImplementedError

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        # No need to reduce.
        if len(output_node.shape) <= 2:
            return output_node

        if self.reduction_type is None:
            reduction_type = hp.Choice(
                REDUCTION_TYPE, [FLATTEN, GLOBAL_MAX, GLOBAL_AVG]
            )
            with hp.conditional_scope(REDUCTION_TYPE, [reduction_type]):
                return self._build_block(hp, output_node, reduction_type)
        else:
            return self._build_block(hp, output_node, self.reduction_type)

    def _build_block(self, hp, output_node, reduction_type):
        if reduction_type == FLATTEN:
            output_node = Flatten().build(hp, output_node)
        elif reduction_type == GLOBAL_MAX:
            output_node = self.global_max(output_node)
        elif reduction_type == GLOBAL_AVG:
            output_node = self.global_avg(output_node)
        return output_node


class SpatialReduction(Reduction):
    """Reduce the dimension of a spatial tensor, e.g. image, to a vector.

    # Arguments
        reduction_type: String. 'flatten', 'global_max' or 'global_avg'.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, reduction_type: Optional[str] = None, **kwargs):
        super().__init__(reduction_type, **kwargs)

    def global_max(self, input_node):
        return layer_utils.get_global_max_pooling(input_node.shape)()(input_node)

    def global_avg(self, input_node):
        return layer_utils.get_global_average_pooling(input_node.shape)()(input_node)


class TemporalReduction(Reduction):
    """Reduce the dimension of a temporal tensor, e.g. output of RNN, to a vector.

    # Arguments
        reduction_type: String. 'flatten', 'global_max' or 'global_avg'. If left
            unspecified, it will be tuned automatically.
    """

    def __init__(self, reduction_type: Optional[str] = None, **kwargs):
        super().__init__(reduction_type, **kwargs)

    def global_max(self, input_node):
        return tf.math.reduce_max(input_node, axis=-2)

    def global_avg(self, input_node):
        return tf.math.reduce_mean(input_node, axis=-2)
