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

from autokeras.engine import serializable


class Preprocessor(serializable.Serializable):
    """A preprocessor for tf.data.Dataset."""

    def fit(self, dataset):
        """Fit the preprocessor with the dataset."""
        raise NotImplementedError

    def transform(self, dataset):
        """Transform the dataset wth the preprocessor."""
        raise NotImplementedError


class TargetPreprocessor(Preprocessor):
    def postprocess(self, dataset):
        raise NotImplementedError
