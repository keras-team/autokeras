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


class TextAdapter(adapter_module.Adapter):
    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError(
                "Expect the data to TextInput to be numpy.ndarray or "
                "tf.data.Dataset, but got {type}.".format(type=type(x))
            )


class StructuredDataAdapter(adapter_module.Adapter):
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


class TimeseriesAdapter(adapter_module.Adapter):
    def __init__(self, lookback=None, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback

    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (pd.DataFrame, np.ndarray, tf.data.Dataset)):
            raise TypeError(
                "Expect the data in TimeseriesInput to be numpy.ndarray"
                " or tf.data.Dataset or pd.DataFrame, but got {type}.".format(
                    type=type(x)
                )
            )

    def convert_to_dataset(self, dataset, batch_size):
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        return super().convert_to_dataset(dataset, batch_size)
