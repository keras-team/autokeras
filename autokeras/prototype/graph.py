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

from tensorflow import keras
from tensorflow import nest

from autokeras import graph
from autokeras.prototype import graph_state


class Graph(graph.Graph):
    def __init__(self, inputs=None, outputs=None, **kwargs):
        super().__init__(inputs, outputs, **kwargs)

    def build(self, hp):
        """Build the HyperModel into a Keras Model."""
        graph_state.init_state()
        self.compile()
        keras_nodes = {}
        keras_input_nodes = []

        # Preparing the inputs of the pipeline.
        for node in self.inputs:
            node_id = self._node_to_id[node]
            input_node = node.build_node(hp)
            output_node = node.build(hp, input_node)
            keras_input_nodes.append(input_node)
            keras_nodes[node_id] = output_node

        # Connecting through the blocks.
        # Don't check the block type to deal with the output since the block has
        # sub blocks of different types. The difference should all be handled in
        # block._build_wrapper().
        for block in self.blocks:
            temp_inputs = [
                keras_nodes[self._node_to_id[input_node]]
                for input_node in block.inputs
            ]
            outputs = block.build(hp, inputs=temp_inputs)
            outputs = nest.flatten(outputs)
            for output_node, real_output_node in zip(block.outputs, outputs):
                keras_nodes[self._node_to_id[output_node]] = real_output_node

        for output_node in self.outputs:
            node = keras_nodes[self._node_to_id[output_node]]
            graph_state.get_state().register_outputs(node)

        model = keras.Model(
            keras_input_nodes,
            [
                keras_nodes[self._node_to_id[output_node]]
                for output_node in self.outputs
            ],
        )
        self._compile_keras_model(hp, model)

        pipeline = None
        return pipeline
