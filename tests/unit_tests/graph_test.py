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

import os

import kerastuner
import pytest

import autokeras as ak
from autokeras import graph as graph_module


def test_input_output_disconnect():
    input_node1 = ak.Input()
    output_node = input_node1
    _ = ak.DenseBlock()(output_node)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    with pytest.raises(ValueError) as info:
        graph_module.Graph(inputs=input_node1, outputs=output_node)
    assert "Inputs and outputs not connected." in str(info.value)


def test_hyper_graph_cycle():
    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    head = ak.RegressionHead()
    output_node = head(output_node)
    head.outputs = output_node1

    with pytest.raises(ValueError) as info:
        graph_module.Graph(inputs=[input_node1, input_node2], outputs=output_node)
    assert "The network has a cycle." in str(info.value)


def test_input_missing():
    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    output_node = ak.RegressionHead()(output_node)

    with pytest.raises(ValueError) as info:
        graph_module.Graph(inputs=input_node1, outputs=output_node)
    assert "A required input is missing for HyperModel" in str(info.value)


def test_graph_basics():
    input_node = ak.Input(shape=(30,))
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead(shape=(1,))(output_node)

    model = graph_module.Graph(inputs=input_node, outputs=output_node).build(
        kerastuner.HyperParameters()
    )
    assert model.input_shape == (None, 30)
    assert model.output_shape == (None, 1)


def test_graph_save_load(tmp_path):
    input1 = ak.Input()
    input2 = ak.Input()
    output1 = ak.DenseBlock()(input1)
    output2 = ak.ConvBlock()(input2)
    output = ak.Merge()([output1, output2])
    output1 = ak.RegressionHead()(output)
    output2 = ak.ClassificationHead()(output)

    graph = graph_module.Graph(
        inputs=[input1, input2],
        outputs=[output1, output2],
    )
    path = os.path.join(tmp_path, "graph")
    graph.save(path)
    graph = graph_module.load_graph(path)

    assert len(graph.inputs) == 2
    assert len(graph.outputs) == 2
    assert isinstance(graph.inputs[0].out_blocks[0], ak.DenseBlock)
    assert isinstance(graph.inputs[1].out_blocks[0], ak.ConvBlock)


def test_merge():
    input_node1 = ak.Input(shape=(30,))
    input_node2 = ak.Input(shape=(40,))
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    output_node = ak.RegressionHead(shape=(1,))(output_node)

    model = graph_module.Graph(
        inputs=[input_node1, input_node2], outputs=output_node
    ).build(kerastuner.HyperParameters())
    assert model.input_shape == [(None, 30), (None, 40)]
    assert model.output_shape == (None, 1)


def test_save_custom_metrics_loss(tmp_path):
    def custom_metric(y_pred, y_true):
        return 1

    def custom_loss(y_pred, y_true):
        return y_pred - y_true

    head = ak.ClassificationHead(
        loss=custom_loss, metrics=["accuracy", custom_metric]
    )
    input_node = ak.Input()
    output_node = head(input_node)
    graph = graph_module.Graph(input_node, output_node)
    path = os.path.join(tmp_path, "graph")
    graph.save(path)
    new_graph = graph_module.load_graph(
        path,
        custom_objects={"custom_metric": custom_metric, "custom_loss": custom_loss},
    )
    assert new_graph.blocks[0].metrics[1](0, 0) == 1
    assert new_graph.blocks[0].loss(3, 2) == 1


def test_cat_to_num_with_img_input_error():
    input_node = ak.ImageInput()
    output_node = ak.CategoricalToNumerical()(input_node)

    with pytest.raises(TypeError) as info:
        graph_module.Graph(input_node, outputs=output_node).compile()

    assert "CategoricalToNumerical can only be used" in str(info.value)


def test_graph_can_init_with_one_missing_output():
    input_node = ak.ImageInput()
    output_node = ak.ConvBlock()(input_node)
    output_node = ak.RegressionHead()(output_node)
    ak.ClassificationHead()(output_node)

    graph_module.Graph(input_node, output_node)
