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

from autokeras.utils import data_utils

INT = "int"
NONE = "none"
ONE_HOT = "one-hot"


class PreprocessingLayer(layers.Layer):
    pass


@keras.utils.register_keras_serializable()
class CastToFloat32(PreprocessingLayer):
    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        # Does not and needs not handle strings.
        return data_utils.cast_to_float32(inputs)

    def adapt(self, data):
        return


@keras.utils.register_keras_serializable()
class ExpandLastDim(PreprocessingLayer):
    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        return ops.expand_dims(inputs, axis=-1)

    def adapt(self, data):
        return
