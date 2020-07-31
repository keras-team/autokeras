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
import tensorflow as tf

from autokeras.engine import serializable
from autokeras.utils import data_utils


class Adapter(serializable.Serializable):
    """Adpat the input and output format for Keras Model.

    Adapter is used by the input nodes and the heads of the hypermodel. It analyzes
    the training data to get useful information, e.g., the shape of the data, which
    is required for building the Keras Model. It also converts the dataset to
    tf.data.Dataset format.

    # Arguments
        shape: Tuple of int. The input or output shape of the hypermodel.
        batch_size: Int. Number of samples per gradient update. Defaults to 32.
    """

    def __init__(self, shape=None, batch_size=32):
        self.shape = shape
        self.batch_size = batch_size

    def check(self, dataset):
        """Check if the dataset is valid for the input node.

        # Arguments
            dataset: Usually numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                The dataset to be checked.

        # Returns
            Boolean. Whether the dataset is valid for the input node.
        """
        return True

    def convert_to_dataset(self, dataset):
        """Convert supported formats of datasets to tf.data.Dataset.

        # Arguments
            dataset: Usually numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                The dataset to be converted.

        # Returns
            tf.data.Dataset. The converted dataset.
        """
        if isinstance(dataset, np.ndarray):
            dataset = tf.data.Dataset.from_tensor_slices(dataset)
        return data_utils.batch_dataset(dataset, self.batch_size)

    def fit(self, dataset):
        """Analyze the dataset and record useful information.

        # Arguments
            dataset: tf.data.Dataset.
        """
        self._record_dataset_shape(dataset)

    def fit_before_convert(self, dataset):
        """Analyze the dataset before converting to tf.data.Dataset.

        # Arguments
            dataset: Usually numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
        """
        pass

    def fit_transform(self, dataset):
        self.check(dataset)
        self.fit_before_convert(dataset)
        dataset = self.convert_to_dataset(dataset)
        self.fit(dataset)
        return dataset

    def _record_dataset_shape(self, dataset):
        self.shape = data_utils.dataset_shape(dataset)[1:].as_list()

    def transform(self, dataset):
        """Transform the input dataset to tf.data.Dataset.

        # Arguments
            dataset: Usually numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                The dataset to be transformed.

        # Returns
            tf.data.Dataset. The converted dataset.
        """
        self.check(dataset)
        return self.convert_to_dataset(dataset)

    def get_config(self):
        return {"shape": self.shape}
