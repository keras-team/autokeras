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

import keras
import numpy as np

from autokeras.engine import preprocessor


@keras.utils.register_keras_serializable(package="autokeras")
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
            dataset: numpy.ndarray. The dataset to be transformed.

        # Returns
            numpy.ndarray. The transformed dataset.
        """
        label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        return np.array(
            [label_to_idx[label] for label in dataset.flatten()]
        ).reshape(dataset.shape)


@keras.utils.register_keras_serializable(package="autokeras")
class OneHotEncoder(Encoder):
    def transform(self, dataset):
        """Transform labels to one-hot encodings.

        # Arguments
            dataset: numpy.ndarray. The dataset to be transformed.

        # Returns
            numpy.ndarray. The transformed dataset.
        """
        dataset = super().transform(dataset)
        eye = np.eye(len(self.labels))
        return eye[np.squeeze(dataset, axis=-1)]

    def postprocess(self, data):
        """Transform probabilities back to labels.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification
                head.

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


@keras.utils.register_keras_serializable(package="autokeras")
class LabelEncoder(Encoder):
    """Transform the labels to integer encodings."""

    def transform(self, dataset):
        """Transform labels to integer encodings.

        # Arguments
            dataset: numpy.ndarray. The dataset to be transformed.

        # Returns
            numpy.ndarray. The transformed dataset.
        """
        dataset = super().transform(dataset)
        return dataset

    def postprocess(self, data):
        """Transform probabilities back to labels.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification
                head.

        # Returns
            numpy.ndarray. The original labels.
        """
        return np.array(
            list(map(lambda x: self.labels[int(round(x[0]))], np.array(data)))
        ).reshape(-1, 1)
