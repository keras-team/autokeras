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

from autokeras.engine import preprocessor


class Encoder(preprocessor.TargetPreprocessor):
    """OneHotEncoder to encode and decode the labels.

    This class provides ways to transform data's classification label into vector.

    # Arguments
        labels: A list of strings.
    """

    def __init__(self, labels, **kwargs):
        super().__init__(**kwargs)
        self.labels = labels

    def get_config(self):
        return {"labels": self.labels}

    def fit(self, dataset):
        pass


class OneHotEncoder(Encoder):
    def transform(self, dataset):
        keys_tensor = tf.constant(self.labels)
        vals_tensor = tf.constant(list(range(len(self.labels))))
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1
        )
        eye = tf.eye(len(self.labels))
        dataset = dataset.map(
            lambda x: tf.nn.embedding_lookup(eye, table.lookup(tf.reshape(x, [-1])))
        )
        return dataset

    def postprocess(self, data):
        """Get label for every element in data.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The original labels.
        """
        return np.array(
            list(
                map(
                    lambda x: self.labels[x],
                    np.argmax(np.array(data), axis=1),
                )
            )
        ).reshape(-1, 1)


class LabelEncoder(Encoder):
    def transform(self, dataset):
        keys_tensor = tf.constant(self.labels)
        vals_tensor = tf.constant(list(range(len(self.labels))))
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1
        )
        return dataset.map(table)

    def postprocess(self, data):
        """Get label for every element in data.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The original labels.
        """
        return np.array(
            list(map(lambda x: self.labels[int(round(x[0]))], np.array(data)))
        ).reshape(-1, 1)


class MultiLabelEncoder(Encoder):
    def __init__(self, **kwargs):
        super().__init__(labels=[], **kwargs)

    def transform(self, dataset):
        return dataset

    def postprocess(self, data):
        data[data < 0.5] = 0
        data[data > 0.5] = 1
        return data
