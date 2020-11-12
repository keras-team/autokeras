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

from autokeras import blocks
from autokeras import nodes


def test_input_get_block_return_general_block():
    input_node = nodes.Input()
    assert isinstance(input_node.get_block(), blocks.GeneralBlock)


def test_time_series_input_node_build_no_error():
    node = nodes.TimeseriesInput(lookback=2, shape=(32,))
    hp = kerastuner.HyperParameters()

    input_node = node.build_node(hp)
    node.build(hp, input_node)


def test_time_series_input_node_deserialize_build_no_error():
    node = nodes.TimeseriesInput(lookback=2, shape=(32,))
    node = nodes.deserialize(nodes.serialize(node))
    hp = kerastuner.HyperParameters()

    input_node = node.build_node(hp)
    node.build(hp, input_node)
