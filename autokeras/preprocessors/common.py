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


class LambdaPreprocessor(preprocessor.Preprocessor):
    """Build Preprocessor with a map function."""

    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def transform(self, dataset):
        return dataset.map(self.func)


class AddOneDimension(LambdaPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(lambda x: tf.expand_dims(x, axis=-1), **kwargs)

    def get_config(self):
        return {}


class CastToInt32(preprocessor.Preprocessor):
    def get_config(self):
        return {}

    def transform(self, dataset):
        return dataset.map(lambda x: tf.cast(x, tf.int32))


class CastToString(preprocessor.Preprocessor):
    def get_config(self):
        return {}

    def transform(self, dataset):
        return dataset.map(tf.strings.as_string)


class SlidingWindow(preprocessor.Preprocessor):
    def __init__(self, lookback, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.batch_size = batch_size

    def transform(self, dataset):
        dataset = dataset.unbatch()
        dataset = dataset.window(self.lookback, shift=1, drop_remainder=True)
        # dataset = dataset.flat_map(lambda x: x.batch(self.lookback))
        # return dataset.batch(self.batch_size)
        final_data = []
        for window in dataset:
            final_data.append([elems.numpy() for elems in window])
        dataset = tf.data.Dataset.from_tensor_slices(final_data)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def get_config(self):
        return {"lookback": self.lookback, "batch_size": self.batch_size}
