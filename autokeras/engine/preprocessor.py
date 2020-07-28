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

import kerastuner

from autokeras.engine import serializable


class Preprocessor(kerastuner.HyperModel, serializable.Serializable):
    """Input data preprocessor search space.

    This class defines the search space for input data preprocessor. A
    preprocessor transforms the dataset using `tf.data` operations.
    """

    def build(self, hp, x):
        """Build the `tf.data` input preprocessor.

        # Arguments
            hp: `HyperParameters` instance. The hyperparameters for building the
                model.
            x: `tf.data.Dataset` instance. The input data for preprocessing.

        # Returns
            `tf.data.Dataset`. The preprocessed data to pass to the model.
        """
        raise NotImplementedError
