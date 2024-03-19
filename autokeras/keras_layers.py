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


@keras.utils.register_keras_serializable()
class WarmUp(keras.optimizers.schedules.LearningRateSchedule):
    """official.nlp.optimization.WarmUp"""

    def __init__(
        self,
        initial_learning_rate,
        decay_schedule_fn,
        warmup_steps,
        power=1.0,
        name=None,
    ):
        super(WarmUp, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with keras.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps,
            # the learning rate will be
            # `global_step/num_warmup_steps * init_lr`.
            global_step_float = ops.cast(step, "float32")
            warmup_steps_float = ops.cast(self.warmup_steps, "float32")
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * ops.power(
                warmup_percent_done, self.power
            )
            return ops.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step),
                name=name,
            )

    def get_config(self):
        return {  # pragma: no cover
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }
