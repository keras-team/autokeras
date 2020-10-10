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

from autokeras.engine import preprocessor


class PostProcessor(preprocessor.TargetPreprocessor):
    def transform(self, dataset):
        return dataset


class SigmoidPostprocessor(PostProcessor):
    """Postprocessor for sigmoid outputs."""

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


class SoftmaxPostprocessor(PostProcessor):
    """Postprocessor for softmax outputs."""

    def postprocess(self, data):
        """Transform probabilities to zeros and ones.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The zeros and ones predictions.
        """
        idx = np.argmax(data, axis=-1)
        data = np.zeros(data.shape)
        data[np.arange(data.shape[0]), idx] = 1
        return data
