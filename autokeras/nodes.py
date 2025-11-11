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

from typing import Dict
from typing import List
from typing import Optional

import keras
import tree

from autokeras import adapters
from autokeras import analysers
from autokeras import blocks
from autokeras import hyper_preprocessors
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import io_hypermodel
from autokeras.engine import node as node_module
from autokeras.utils import utils


def serialize(obj):
    return utils.serialize_keras_object(obj)


def deserialize(config, custom_objects=None):
    return utils.deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
    )


@keras.utils.register_keras_serializable(package="autokeras")
class Input(node_module.Node, io_hypermodel.IOHyperModel):
    """Input node for tensor data.

    The data should be numpy.ndarray.

    # Arguments
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build_node(self, hp):
        return keras.Input(shape=self.shape, dtype=self.dtype)

    def build(self, hp, inputs=None):
        input_node = tree.flatten(inputs)[0]
        return keras_layers.CastToFloat32()(input_node)

    def get_adapter(self):
        return adapters.InputAdapter()

    def get_analyser(self):
        return analysers.InputAnalyser()

    def get_block(self):
        return blocks.GeneralBlock()

    def get_hyper_preprocessors(self):
        return []


class ImageInput(Input):
    """Input node for image data.

    The input data should be numpy.ndarray. The shape of the
    data should be should be (samples, width, height) or (samples, width,
    height, channels).

    # Arguments
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, hp, inputs=None):
        inputs = super().build(hp, inputs)
        output_node = tree.flatten(inputs)[0]
        if len(output_node.shape) == 3:
            output_node = keras_layers.ExpandLastDim()(output_node)
        return output_node

    def get_adapter(self):
        return adapters.ImageAdapter()

    def get_analyser(self):
        return analysers.ImageAnalyser()

    def get_block(self):
        return blocks.ImageBlock()


class TextInput(Input):
    """Input node for text data.

    The input data should be numpy.ndarray. The data should be
    one-dimensional. Each element in the data should be a string which is a
    full sentence.

    # Arguments
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build_node(self, hp):
        return keras.Input(shape=self.shape, dtype="int32")

    def build(self, hp, inputs=None):
        output_node = tree.flatten(inputs)[0]
        if len(output_node.shape) == 1:
            output_node = keras_layers.ExpandLastDim()(  # pragma: no cover
                output_node
            )
        return output_node

    def get_adapter(self):
        return adapters.TextAdapter()

    def get_analyser(self):
        return analysers.TextAnalyser()

    def get_block(self):
        return blocks.TextBlock()

    def get_hyper_preprocessors(self):
        return [
            hyper_preprocessors.DefaultHyperPreprocessor(
                preprocessors.CastToString()
            ),
            hyper_preprocessors.DefaultHyperPreprocessor(
                preprocessors.TextTokenizer()
            ),
        ]


class StructuredDataInput(Input):
    """Input node for structured data.

    The input data should be numpy.ndarray. The data should be two-dimensional
    with numerical or categorical values.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the
            data. Defaults to None.
        column_types: Dict. The keys are the column names. The values should
            either be 'numerical' or 'categorical', indicating the type of that
            column. Defaults to None. If not None, the column_names need to be
            specified. If None, it will be inferred from the data. A column will
            be judged as categorical if the number of different values is less
            than 5% of the number of instances.
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(
        self,
        column_names: Optional[List[str]] = None,
        column_types: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.column_names = column_names
        self.column_types = column_types

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "column_names": self.column_names,
                "column_types": self.column_types,
            }
        )
        return config

    def get_adapter(self):
        return adapters.StructuredDataAdapter()

    def get_analyser(self):
        return analysers.StructuredDataAnalyser(
            self.column_names, self.column_types
        )

    def get_block(self):
        return blocks.StructuredDataBlock()

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self.column_names = analyser.column_names
        # Analyser keeps the specified ones and infer the missing ones.
        self.column_types = analyser.column_types

    def build(self, hp, inputs=None):
        return inputs

    def get_hyper_preprocessors(self):
        return [
            hyper_preprocessors.DefaultHyperPreprocessor(
                preprocessors.CategoricalToNumerical(
                    column_names=self.column_names,
                    column_types=self.column_types,
                )
            )
        ]
