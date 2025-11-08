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
from autokeras.utils import data_utils


class LambdaPreprocessor(preprocessor.Preprocessor):
    """Build Preprocessor with a map function.

    # Arguments
        func: a callable function for the dataset to map.
    """

    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def transform(self, dataset):
        return dataset.map(self.func)


class AddOneDimension(LambdaPreprocessor):
    """Append one dimension of size one to the dataset shape."""

    def __init__(self, **kwargs):
        super().__init__(lambda x: np.expand_dims(x, axis=-1), **kwargs)


class CastToInt32(preprocessor.Preprocessor):
    """Cast the dataset shape to int32."""

    def transform(self, dataset):
        return dataset.map(lambda x: x.astype("int32"))


class CastToString(preprocessor.Preprocessor):
    """Cast the dataset shape to string."""

    def transform(self, dataset):
        return dataset.map(data_utils.cast_to_string)
