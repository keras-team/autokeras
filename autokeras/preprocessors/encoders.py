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
    """Transform labels to encodings.

    # Arguments
        labels: A list of labels of any type. The labels to be encoded.
    """

    def __init__(self, labels, **kwargs):
        super().__init__(**kwargs)
        self.labels = [
            label.decode("utf-8") if isinstance(label, bytes) else str(label)
            for label in labels
        ]

    def get_config(self):
        return {"labels": self.labels}

    def fit(self, dataset):
        return

    def transform(self, dataset):
        """Transform labels to integer encodings.

        # Arguments
            dataset: tf.data.Dataset. The dataset to be transformed.

        # Returns
            tf.data.Dataset. The transformed dataset.
        """
        keys_tensor = tf.constant(self.labels)
        vals_tensor = tf.constant(list(range(len(self.labels))))
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1
        )

        return dataset.map(lambda x: table.lookup(tf.reshape(x, [-1])))


class OneHotEncoder(Encoder):
    def transform(self, dataset):
        """Transform labels to one-hot encodings.

        # Arguments
            dataset: tf.data.Dataset. The dataset to be transformed.

        # Returns
            tf.data.Dataset. The transformed dataset.
        """
        dataset = super().transform(dataset)
        eye = tf.eye(len(self.labels))
        dataset = dataset.map(lambda x: tf.nn.embedding_lookup(eye, x))
        return dataset

    def postprocess(self, data):
        """Transform probabilities back to labels.

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
    """Transform the labels to integer encodings."""

    def transform(self, dataset):
        """Transform labels to integer encodings.

        # Arguments
            dataset: tf.data.Dataset. The dataset to be transformed.

        # Returns
            tf.data.Dataset. The transformed dataset.
        """
        dataset = super().transform(dataset)
        dataset = dataset.map(lambda x: tf.expand_dims(x, axis=-1))
        return dataset

    def postprocess(self, data):
        """Transform probabilities back to labels.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The original labels.
        """
        return np.array(
            list(map(lambda x: self.labels[int(round(x[0]))], np.array(data)))
        ).reshape(-1, 1)


class MultiLabelEncoder(Encoder):
    """Encoder for multi-label data."""

    def __init__(self, **kwargs):
        super().__init__(labels=[], **kwargs)

    def transform(self, dataset):
        return dataset

    def postprocess(self, data):
        """Transform probabilities to zeros and ones.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The zeros and ones predictions.
        """
        data[data < 0.5] = 0
        data[data > 0.5] = 1
        return data
