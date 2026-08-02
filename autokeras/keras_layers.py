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
from keras import layers
from keras import ops
import numpy as np

from autokeras.utils import data_utils

INT = "int"
NONE = "none"
ONE_HOT = "one-hot"


class PreprocessingLayer(layers.Layer):
    pass


@keras.utils.register_keras_serializable(package="autokeras")
class CastToFloat32(PreprocessingLayer):
    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        # Does not and needs not handle strings.
        return data_utils.cast_to_float32(inputs)

    def adapt(self, data):
        return


@keras.utils.register_keras_serializable(package="autokeras")
class ExpandLastDim(PreprocessingLayer):
    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        return ops.expand_dims(inputs, axis=-1)

    def adapt(self, data):
        return


@keras.utils.register_keras_serializable(package="autokeras")
class CategoricalToNumericalLayer(PreprocessingLayer):
    """Keras layer that encodes categorical features to numerical integers.

    This layer is the model-graph counterpart of the pipeline preprocessor
    CategoricalToNumerical. It allows export_model() to produce a self-contained
    Keras model that accepts raw structured data (including categorical columns)
    and produces the same predictions as AutoModel.predict/evaluate.

    # Arguments
        column_names: list of str. Names of the columns.
        column_types: dict. Mapping column name -> "categorical" or "numerical".
        encoding: list of str. Per-column encoding type ("int" or "none").
        vocabulary: list of list. For each categorical column, the list of
            known category values (as strings). For numerical columns use None.
    """

    def __init__(
        self,
        column_names=None,
        column_types=None,
        encoding=None,
        vocabulary=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.column_names = column_names or []
        self.column_types = column_types or {}
        self.encoding = encoding or []
        self.vocabulary = vocabulary or []
        # Built lookup tables (list of dict or None)
        self._lookups = []

    def build(self, input_shape):
        self._lookups = []
        for i, enc in enumerate(self.encoding):
            if enc == "none" or not self.vocabulary or self.vocabulary[i] is None:
                self._lookups.append(None)
            else:
                vocab = self.vocabulary[i]
                # Map string category -> integer index
                lookup = {str(v): idx for idx, v in enumerate(vocab)}
                self._lookups.append(lookup)
        super().build(input_shape)

    def call(self, inputs):
        # inputs is expected to be a float or object tensor of shape (batch, n_features)
        # We convert to numpy for the lookup (works for TF eager / numpy backend;
        # for full multi-backend graph support a pure-ops implementation would be needed).
        # Because AutoKeras 3 primarily targets eager inference for exported models,
        # this is acceptable and matches the original preprocessor behaviour.
        x = ops.convert_to_numpy(inputs) if hasattr(ops, "convert_to_numpy") else np.asarray(inputs)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        outputs = []
        for i, enc in enumerate(self.encoding):
            col = x[:, i : i + 1]
            if enc == "none" or self._lookups[i] is None:
                col = col.astype("float32")
                col = np.where(np.isnan(col), 0.0, col)
                outputs.append(col)
            else:
                lookup = self._lookups[i]
                # Encode each value; unknown -> len(vocab)
                encoded = []
                for val in col.flatten():
                    key = str(val)
                    encoded.append(lookup.get(key, len(lookup)))
                encoded = np.array(encoded, dtype="float32").reshape(col.shape)
                outputs.append(encoded)
        result = np.concatenate(outputs, axis=1).astype("float32")
        return ops.convert_to_tensor(result)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "column_names": self.column_names,
                "column_types": self.column_types,
                "encoding": self.encoding,
                "vocabulary": self.vocabulary,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def from_preprocessor(cls, preprocessor):
        """Build a fitted layer from a CategoricalToNumerical preprocessor instance."""
        vocabulary = []
        for enc, encoder in zip(preprocessor.encoding, preprocessor.encoders):
            if enc == "none" or encoder is None:
                vocabulary.append(None)
            else:
                # encoder.labels is the list of known categories
                vocabulary.append(list(encoder.labels))
        return cls(
            column_names=preprocessor.column_names,
            column_types=preprocessor.column_types,
            encoding=preprocessor.encoding,
            vocabulary=vocabulary,
        )
