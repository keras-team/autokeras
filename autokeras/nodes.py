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

import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import adapters
from autokeras import analysers
from autokeras import blocks
from autokeras import hyper_preprocessors as hpps_module
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import io_hypermodel
from autokeras.engine import node as node_module


def serialize(obj):
    return tf.keras.utils.serialize_keras_object(obj)


def deserialize(config, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="nodes",
    )


class Input(node_module.Node, io_hypermodel.IOHyperModel):
    """Input node for tensor data.

    The data should be numpy.ndarray or tf.data.Dataset.

    # Arguments
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build_node(self, hp):
        return tf.keras.Input(shape=self.shape, dtype=self.dtype)

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
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

    The input data should be numpy.ndarray or tf.data.Dataset. The shape of the data
    should be should be (samples, width, height) or
    (samples, width, height, channels).

    # Arguments
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, hp, inputs=None):
        inputs = super().build(hp, inputs)
        output_node = nest.flatten(inputs)[0]
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

    The input data should be numpy.ndarray or tf.data.Dataset. The data should be
    one-dimensional. Each element in the data should be a string which is a full
    sentence.

    # Arguments
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build_node(self, hp):
        return tf.keras.Input(shape=self.shape, dtype=tf.string)

    def build(self, hp, inputs=None):
        output_node = nest.flatten(inputs)[0]
        if len(output_node.shape) == 1:
            output_node = keras_layers.ExpandLastDim()(output_node)
        return output_node

    def get_adapter(self):
        return adapters.TextAdapter()

    def get_analyser(self):
        return analysers.TextAnalyser()

    def get_block(self):
        return blocks.TextBlock()


class StructuredDataInput(Input):
    """Input node for structured data.

    The input data should be numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
    The data should be two-dimensional with numerical or categorical values.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will be obtained from the header of the csv
            file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data. A column will be judged as
            categorical if the number of different values is less than 5% of the
            number of instances.
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
            {"column_names": self.column_names, "column_types": self.column_types}
        )
        return config

    def get_adapter(self):
        return adapters.StructuredDataAdapter()

    def get_analyser(self):
        return analysers.StructuredDataAnalyser(self.column_names, self.column_types)

    def get_block(self):
        return blocks.StructuredDataBlock()

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self.column_names = analyser.column_names
        # Analyser keeps the specified ones and infer the missing ones.
        self.column_types = analyser.column_types

    def build(self, hp, inputs=None):
        return inputs


class TimeseriesInput(StructuredDataInput):
    """Input node for timeseries data.

    # Arguments
        lookback: Int. The range of history steps to consider for each prediction.
            For example, if lookback=n, the data in the range of [i - n, i - 1]
            is used to predict the value of step i. If unspecified, it will be tuned
            automatically.
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will be obtained from the header of the csv
            file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data. A column will be judged as
            categorical if the number of different values is less than 5% of the
            number of instances.
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(
        self,
        lookback: Optional[int] = None,
        column_names: Optional[List[str]] = None,
        column_types: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            column_names=column_names, column_types=column_types, name=name, **kwargs
        )
        self.lookback = lookback

    def get_config(self):
        config = super().get_config()
        config.update({"lookback": self.lookback})
        return config

    def get_adapter(self):
        return adapters.TimeseriesAdapter()

    def get_analyser(self):
        return analysers.TimeseriesAnalyser(
            column_names=self.column_names, column_types=self.column_types
        )

    def get_block(self):
        return blocks.TimeseriesBlock()

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []
        hyper_preprocessors.append(
            hpps_module.DefaultHyperPreprocessor(
                preprocessors.SlidingWindow(
                    lookback=self.lookback, batch_size=self.batch_size
                )
            )
        )
        return hyper_preprocessors
