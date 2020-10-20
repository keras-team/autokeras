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

import tensorflow as tf

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

    def get_config(self):
        return {}


class AddOneDimension(LambdaPreprocessor):
    """Append one dimension of size one to the dataset shape."""

    def __init__(self, **kwargs):
        super().__init__(lambda x: tf.expand_dims(x, axis=-1), **kwargs)


class CastToInt32(preprocessor.Preprocessor):
    """Cast the dataset shape to tf.int32."""

    def get_config(self):
        return {}

    def transform(self, dataset):
        return dataset.map(lambda x: tf.cast(x, tf.int32))


class CastToString(preprocessor.Preprocessor):
    """Cast the dataset shape to tf.string."""

    def get_config(self):
        return {}

    def transform(self, dataset):
        return dataset.map(data_utils.cast_to_string)


class SlidingWindow(preprocessor.Preprocessor):
    """Apply sliding window to the dataset.

    It groups the consecutive data items together. Therefore, it inserts one
    more dimension of size `lookback` to the dataset shape after the batch_size
    dimension. It also reduce the number of instances in the dataset by
    (lookback - 1).

    # Arguments
        lookback: Int. The window size. The number of data items to group
            together.
        batch_size: Int. The batch size of the dataset.
    """

    def __init__(self, lookback, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.batch_size = batch_size

    def transform(self, dataset):
        dataset = dataset.unbatch()
        dataset = dataset.window(self.lookback, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(
            lambda x: x.batch(self.lookback, drop_remainder=True)
        )
        dataset = dataset.batch(self.batch_size)
        return dataset

    def get_config(self):
        return {"lookback": self.lookback, "batch_size": self.batch_size}
