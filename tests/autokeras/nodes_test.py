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

from autokeras import nodes
from autokeras import blocks
from autokeras import preprocessors

def test_input_get_block_return_general_block():
    input_node = nodes.Input()
    assert isinstance(input_node.get_block(), blocks.GeneralBlock)


def test_structured_data_input_get_pps_cast_to_string():
    input_node = nodes.StructuredDataInput()
    input_node.dtype = tf.float32
    assert isinstance(input_node.get_hyper_preprocessors()[0].preprocessor, preprocessors.CastToString)
