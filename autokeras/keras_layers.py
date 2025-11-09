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

from typing import List

import keras
from keras import layers
from keras import ops

from autokeras.utils import data_utils

INT = "int"
NONE = "none"
ONE_HOT = "one-hot"


class PreprocessingLayer(layers.Layer):
    pass


@keras.utils.register_keras_serializable()
class CastToFloat32(PreprocessingLayer):
    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        # Does not and needs not handle strings.
        return data_utils.cast_to_float32(inputs)

    def adapt(self, data):
        return


@keras.utils.register_keras_serializable()
class ExpandLastDim(PreprocessingLayer):
    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        return ops.expand_dims(inputs, axis=-1)

    def adapt(self, data):
        return


@keras.utils.register_keras_serializable()
class MultiCategoryEncoding(PreprocessingLayer):
    """Encode the categorical features to numerical features.

    # Arguments
        encoding: A list of strings, which has the same number of elements as
            the columns in the structured data. Each of the strings specifies
            the encoding method used for the corresponding column. Use 'int' for
            categorical columns and 'none' for numerical columns.
    """

    # TODO: Support one-hot encoding.
    # TODO: Support frequency encoding.

    def __init__(self, encoding: List[str], **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.encoding_layers = []
        for encoding in self.encoding:
            if encoding == NONE:
                self.encoding_layers.append(None)
            elif encoding == INT:
                # Set a temporary vocabulary to prevent the error of no
                # vocabulary when calling the layer to build the model.  The
                # vocabulary would be reset by adapting the layer later.
                self.encoding_layers.append(layers.StringLookup())
            elif encoding == ONE_HOT:
                self.encoding_layers.append(None)

    def build(self, input_shape):
        for encoding_layer in self.encoding_layers:
            if encoding_layer is not None:
                encoding_layer.build(tf.TensorShape([1]))

    def call(self, inputs):
        input_nodes = nest.flatten(inputs)[0]
        split_inputs = tf.split(input_nodes, [1] * len(self.encoding), axis=-1)
        output_nodes = []
        for input_node, encoding_layer in zip(
            split_inputs, self.encoding_layers
        ):
            if encoding_layer is None:
                number = data_utils.cast_to_float32(input_node)
                # Replace NaN with 0.
                imputed = tf.where(
                    tf.math.is_nan(number), tf.zeros_like(number), number
                )
                output_nodes.append(imputed)
            else:
                output_nodes.append(
                    data_utils.cast_to_float32(
                        encoding_layer(data_utils.cast_to_string(input_node))
                    )
                )
        if len(output_nodes) == 1:
            return output_nodes[0]
        return layers.Concatenate()(output_nodes)

    def adapt(self, data):
        for index, encoding_layer in enumerate(self.encoding_layers):
            if encoding_layer is None:
                continue
            data_column = data.map(lambda x: tf.slice(x, [0, index], [-1, 1]))
            encoding_layer.adapt(data_column.map(data_utils.cast_to_string))

    def get_config(self):
        config = {
            "encoding": self.encoding,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
