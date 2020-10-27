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

import tensorflow as tf

from autokeras import analysers
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import preprocessor
from autokeras.utils import data_utils


class LambdaPreprocessor(preprocessor.Preprocessor):
    """Build Preprocessor with a map function.

    # Arguments
        func: a callable function for the dataset to map.
    """

    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def transform(self, dataset):
        return dataset.map(self.func)

    def get_config(self):
        return {}


class AddOneDimension(LambdaPreprocessor):
    """Append one dimension of size one to the dataset shape."""

    def __init__(self, **kwargs):
        super().__init__(lambda x: tf.expand_dims(x, axis=-1), **kwargs)


class CastToInt32(preprocessor.Preprocessor):
    """Cast the dataset shape to tf.int32."""

    def get_config(self):
        return {}

    def transform(self, dataset):
        return dataset.map(lambda x: tf.cast(x, tf.int32))


class CastToString(preprocessor.Preprocessor):
    """Cast the dataset shape to tf.string."""

    def get_config(self):
        return {}

    def transform(self, dataset):
        return dataset.map(data_utils.cast_to_string)


class SlidingWindow(preprocessor.Preprocessor):
    """Apply sliding window to the dataset.

    It groups the consecutive data items together. Therefore, it inserts one
    more dimension of size `lookback` to the dataset shape after the batch_size
    dimension. It also reduce the number of instances in the dataset by
    (lookback - 1).

    # Arguments
        lookback: Int. The window size. The number of data items to group
            together.
        batch_size: Int. The batch size of the dataset.
    """

    def __init__(self, lookback, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.batch_size = batch_size

    def transform(self, dataset):
        dataset = dataset.unbatch()
        dataset = dataset.window(self.lookback, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(
            lambda x: x.batch(self.lookback, drop_remainder=True)
        )
        dataset = dataset.batch(self.batch_size)
        return dataset

    def get_config(self):
        return {"lookback": self.lookback, "batch_size": self.batch_size}


class CategoricalToNumericalPreprocessor(preprocessor.Preprocessor):
    """Encode the categorical features to numerical features.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will be obtained from the header of the csv
            file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data.
    """

    def __init__(self, column_names, column_types, **kwargs):
        super().__init__(**kwargs)
        self.column_names = column_names
        self.column_types = column_types
        encoding = []
        for column_name in self.column_names:
            column_type = self.column_types[column_name]
            if column_type == analysers.CATEGORICAL:
                # TODO: Search to use one-hot or int.
                encoding.append(keras_layers.INT)
            else:
                encoding.append(keras_layers.NONE)
        self.layer = keras_layers.MultiCategoryEncoding(encoding)

    def fit(self, dataset):
        self.layer.adapt(dataset)

    def transform(self, dataset):
        for data in dataset.map(self.layer):
            result = data
        return result

    def get_config(self):
        vocab = []
        for encoding_layer in self.layer.encoding_layers:
            if encoding_layer is None:
                vocab.append([])
            else:
                vocab.append(encoding_layer.get_vocabulary())
        return {
            "column_types": self.column_types,
            "column_names": self.column_names,
            "encoding_layer": preprocessors.serialize(self.layer),
            "encoding_vocab": vocab,
        }

    @classmethod
    def from_config(cls, config):
        init_config = {
            "column_types": config["column_types"],
            "column_names": config["column_names"],
        }
        obj = cls(**init_config)
        obj.layer = preprocessors.deserialize(config["encoding_layer"])
        for encoding_layer, vocab in zip(
            obj.layer.encoding_layers, config["encoding_vocab"]
        ):
            if encoding_layer is not None:
                encoding_layer.set_vocabulary(vocab)
        return obj
