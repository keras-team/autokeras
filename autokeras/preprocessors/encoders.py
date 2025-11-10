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

from autokeras import analysers
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
            [
                label_to_idx.get(str(label), len(self.labels))
                for label in dataset.flatten()
            ]
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
        encoded_int = dataset.squeeze(axis=-1)
        result = np.zeros((len(encoded_int), len(self.labels)))
        for i, idx in enumerate(encoded_int):
            if idx < len(self.labels):
                result[i] = eye[idx]
        return result

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
                    lambda x: (
                        self.labels[x] if x < len(self.labels) else "unknown"
                    ),
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
            list(
                map(
                    lambda x: (
                        self.labels[int(round(x[0]))]
                        if int(round(x[0])) < len(self.labels)
                        else "unknown"
                    ),
                    np.array(data),
                )
            )
        ).reshape(-1, 1)


@keras.utils.register_keras_serializable()
class CategoricalToNumerical(preprocessor.Preprocessor):
    """Encode the categorical features to numerical features.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the
            data.
        column_types: Dict. The keys are the column names. The values should
            either be 'numerical' or 'categorical', indicating the type of that
            column. Defaults to None. If not None, the column_names need to be
            specified.  If None, it will be inferred from the data.
    """

    # TODO: Support one-hot encoding.
    # TODO: Support frequency encoding.

    def __init__(self, column_names, column_types, **kwargs):
        super().__init__(**kwargs)
        self.column_names = column_names
        self.column_types = column_types
        self.encoding = []
        for column_name in self.column_names:
            column_type = self.column_types[column_name]
            if column_type == analysers.CATEGORICAL:
                # TODO: Search to use one-hot or int.
                self.encoding.append("int")
            else:
                self.encoding.append("none")
        self.encoders = [None] * len(self.encoding)

    def fit(self, dataset):
        for i, enc_type in enumerate(self.encoding):
            if enc_type == "none":
                continue
            column_data = dataset[:, i]
            unique_labels = np.unique(column_data)
            if enc_type == "int":
                self.encoders[i] = LabelEncoder(unique_labels)
            elif enc_type == "one_hot":  # pragma: no cover
                self.encoders[i] = OneHotEncoder(  # pragma: no cover
                    unique_labels
                )
            else:
                raise ValueError(  # pragma: no cover
                    f"Unsupported encoding: {enc_type}"
                )

    def transform(self, dataset):
        outputs = []
        for i, enc_type in enumerate(self.encoding):
            column_data = dataset[:, i : i + 1]
            if enc_type == "none":
                column_data = column_data.astype("float32")
                # Replace NaN with 0.
                imputed = np.where(np.isnan(column_data), 0, column_data)
                outputs.append(imputed)
            else:
                encoded = self.encoders[i].transform(column_data)
                outputs.append(encoded)
        return np.concatenate(outputs, axis=1).astype("float32")

    def get_config(self):
        encoders_config = []
        for enc in self.encoders:
            if enc is None:
                encoders_config.append(
                    {"encoder_type": None, "encoder_config": None}
                )
            else:
                encoder_type = (
                    "label" if isinstance(enc, LabelEncoder) else "one_hot"
                )
                encoders_config.append(
                    {
                        "encoder_type": encoder_type,
                        "encoder_config": enc.get_config(),
                    }
                )
        return {
            "column_types": self.column_types,
            "column_names": self.column_names,
            "encoding": self.encoding,
            "encoders": encoders_config,
        }

    @classmethod
    def from_config(cls, config):
        obj = cls(config["column_names"], config["column_types"])
        obj.encoding = config["encoding"]
        obj.encoders = []
        for item in config["encoders"]:
            if item["encoder_type"] is None:
                obj.encoders.append(None)
            elif item["encoder_type"] == "label":
                obj.encoders.append(LabelEncoder(**item["encoder_config"]))
            elif item["encoder_type"] == "one_hot":  # pragma: no cover
                obj.encoders.append(  # pragma: no cover
                    OneHotEncoder(**item["encoder_config"])
                )
        return obj
