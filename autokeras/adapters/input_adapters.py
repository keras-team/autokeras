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

import numpy as np
import pandas as pd
import tensorflow as tf

from autokeras.engine import adapter as adapter_module
from autokeras.utils import data_utils


class InputAdapter(adapter_module.Adapter):
    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError(
                "Expect the data to Input to be numpy.ndarray or "
                "tf.data.Dataset, but got {type}.".format(type=type(x))
            )
        if isinstance(x, np.ndarray) and not np.issubdtype(x.dtype, np.number):
            raise TypeError(
                "Expect the data to Input to be numerical, but got "
                "{type}.".format(type=x.dtype)
            )


class ImageAdapter(adapter_module.Adapter):
    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError(
                "Expect the data to ImageInput to be numpy.ndarray or "
                "tf.data.Dataset, but got {type}.".format(type=type(x))
            )
        if isinstance(x, np.ndarray) and not np.issubdtype(x.dtype, np.number):
            raise TypeError(
                "Expect the data to ImageInput to be numerical, but got "
                "{type}.".format(type=x.dtype)
            )


class TextInputAdapter(adapter_module.Adapter):
    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError(
                "Expect the data to TextInput to be numpy.ndarray or "
                "tf.data.Dataset, but got {type}.".format(type=type(x))
            )

        if isinstance(x, np.ndarray) and x.ndim != 1:
            raise ValueError(
                "Expect the data to TextInput to have 1 dimension, but "
                "got input shape {shape} with {ndim} dimensions".format(
                    shape=x.shape, ndim=x.ndim
                )
            )
        if isinstance(x, np.ndarray) and not np.issubdtype(x.dtype, np.character):
            raise TypeError(
                "Expect the data to TextInput to be strings, but got "
                "{type}.".format(type=x.dtype)
            )

    def convert_to_dataset(self, x):
        x = super().convert_to_dataset(x)
        shape = data_utils.dataset_shape(x)
        if len(shape) == 1:
            x = x.map(lambda a: tf.reshape(a, [-1, 1]))
        return x


class StructuredDataInputAdapter(adapter_module.Adapter):
    def check(self, x):
        if not isinstance(x, (pd.DataFrame, np.ndarray, tf.data.Dataset)):
            raise TypeError(
                "Unsupported type {type} for "
                "{name}.".format(type=type(x), name=self.__class__.__name__)
            )

    def convert_to_dataset(self, dataset, batch_size):
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        if isinstance(dataset, np.ndarray) and dataset.dtype == np.object:
            dataset = dataset.astype(np.unicode)
        return super().convert_to_dataset(dataset, batch_size)


class TimeseriesInputAdapter(adapter_module.Adapter):
    def __init__(
        self, lookback=None, column_names=None, column_types=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.column_names = column_names
        self.column_types = column_types

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lookback": self.lookback,
                "column_names": self.column_names,
                "column_types": self.column_types,
            }
        )
        return config

    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (pd.DataFrame, np.ndarray, tf.data.Dataset)):
            raise TypeError(
                "Expect the data in TimeseriesInput to be numpy.ndarray"
                " or tf.data.Dataset or pd.DataFrame, but got {type}.".format(
                    type=type(x)
                )
            )

        if isinstance(x, np.ndarray) and x.ndim != 2:
            raise ValueError(
                "Expect the data in TimeseriesInput to have 2 dimension"
                ", but got input shape {shape} with {ndim} "
                "dimensions".format(shape=x.shape, ndim=x.ndim)
            )

    def convert_to_dataset(self, x):
        if isinstance(x, pd.DataFrame):
            # Convert x, y, validation_data to tf.Dataset.
            x = x.values.astype(np.float32)
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
        x = tf.data.Dataset.from_tensor_slices(x)
        x = x.window(self.lookback, shift=1, drop_remainder=True)
        final_data = []
        for window in x:
            final_data.append([elems.numpy() for elems in window])
        final_data = tf.data.Dataset.from_tensor_slices(final_data)
        return super().convert_to_dataset(final_data)
