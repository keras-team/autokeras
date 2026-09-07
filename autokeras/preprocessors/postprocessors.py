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
class PostProcessor(preprocessor.TargetPreprocessor):
    def transform(self, dataset):
        return dataset


@keras.utils.register_keras_serializable(package="autokeras")
class SigmoidPostprocessor(PostProcessor):
    """Postprocessor for sigmoid outputs."""

    def postprocess(self, data):
        """Transform probabilities to zeros and ones.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification
                head.

        # Returns
            numpy.ndarray. The zeros and ones predictions.
        """
        data[data < 0.5] = 0
        data[data > 0.5] = 1
        return data


@keras.utils.register_keras_serializable(package="autokeras")
class SoftmaxPostprocessor(PostProcessor):
    """Postprocessor for softmax outputs."""

    def postprocess(self, data):
        """Transform probabilities to zeros and ones.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification
                head.

        # Returns
            numpy.ndarray. The zeros and ones predictions.
        """
        idx = np.argmax(data, axis=-1)
        data = np.zeros(data.shape)
        data[np.arange(data.shape[0]), idx] = 1
        return data


@keras.utils.register_keras_serializable(package="autokeras")
class TargetNormalizer(preprocessor.TargetPreprocessor):
    """Standardize regression targets and invert them at predict time.

    AutoKeras regression heads use an unbounded Dense output. On small
    tabular datasets with mixed-scale features this can produce predictions
    far outside the observed target range. Training on z-scored targets and
    inverting in ``postprocess`` keeps predictions numerically plausible
    even when the model is weak (issue #1964).
    """

    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.mean = self._to_array(mean)
        self.std = self._to_array(std)

    @staticmethod
    def _to_array(value):
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32)

    def fit(self, dataset):
        data = np.asarray(dataset, dtype=np.float32)
        self.mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # Constant targets would otherwise divide by zero.
        self.std = np.where(std < 1e-7, 1.0, std).astype(np.float32)

    def transform(self, dataset):
        data = np.asarray(dataset, dtype=np.float32)
        if self.mean is None or self.std is None:
            return data
        return (data - self.mean) / self.std

    def postprocess(self, data):
        """Map model outputs from z-score space back to the target scale."""
        data = np.asarray(data, dtype=np.float32)
        if self.mean is None or self.std is None:
            return data
        return data * self.std + self.mean

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mean": None if self.mean is None else self.mean.tolist(),
                "std": None if self.std is None else self.std.tolist(),
            }
        )
        return config
