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

from autokeras.prototype import base_block
from autokeras.prototype import graph_state


def is_dataset(inputs):
    raise NotImplementedError


def convert_to_keras_inputs(inputs):
    raise NotImplementedError


class Block(base_block.BaseBlock):
    # Open for extension. The same as the old Block class.
    def _build_wrapper(self, hp, inputs, *args, **kwargs):
        # Accept only KerasTensor.
        if is_dataset(inputs):
            inputs = convert_to_keras_inputs(inputs)
        graph_state.get_state().register_inputs(inputs)
        return super()._build_wrapper(hp, inputs, *args, **kwargs)
