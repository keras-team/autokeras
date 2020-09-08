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

from autokeras.engine import named_hypermodel


class HyperPreprocessor(named_hypermodel.NamedHyperModel):
    """Input data preprocessor search space.

    This class defines the search space for a Preprocessor.
    """

    def build(self, hp, dataset):
        """Build the `tf.data` input preprocessor.

        # Arguments
            hp: `HyperParameters` instance. The hyperparameters for building the
                a Preprocessor.
            dataset: tf.data.Dataset.

        # Returns
            an instance of Preprocessor.
        """
        raise NotImplementedError
