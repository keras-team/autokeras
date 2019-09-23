from unittest import mock

import numpy as np
import pytest

import autokeras as ak
from tests import common


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


def test_graph_auto_model_basic(tmp_dir):
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100, 1)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    graph = ak.GraphAutoModel(input_node,
                              output_node,
                              directory=tmp_dir,
                              max_trials=1)
    graph.fit(x_train, y_train, epochs=1, validation_data=(x_train, y_train))
    result = graph.predict(x_train)

    assert result.shape == (100, 1)


def test_merge(tmp_dir):
    x_train = np.random.rand(100, 33)
    y_train = np.random.rand(100, 1)

    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    output_node = ak.RegressionHead()(output_node)

    graph = ak.GraphAutoModel([input_node1, input_node2],
                              output_node,
                              directory=tmp_dir,
                              max_trials=1)
    graph.fit([x_train, x_train], y_train,
              epochs=1,
              batch_size=100,
              verbose=False,
              validation_split=0.5)
    result = graph.predict([x_train, x_train])

    assert result.shape == (100, 1)


@mock.patch('kerastuner.engine.tuner.Tuner.search',
            side_effect=lambda *args, **kwargs: None)
def test_preprocessing(_, tmp_dir):
    x_train = np.random.rand(100, 33)
    y_train = np.random.rand(100, 1)

    input_node1 = ak.Input()
    temp_node1 = ak.Normalization()(input_node1)
    output_node1 = ak.DenseBlock()(temp_node1)

    output_node3 = ak.Normalization()(temp_node1)
    output_node3 = ak.DenseBlock()(output_node3)

    input_node2 = ak.Input()
    output_node2 = ak.Normalization()(input_node2)
    output_node2 = ak.DenseBlock()(output_node2)

    output_node = ak.Merge()([output_node1, output_node2, output_node3])
    output_node = ak.RegressionHead()(output_node)

    graph = ak.GraphAutoModel([input_node1, input_node2],
                              output_node,
                              directory=tmp_dir,
                              max_trials=1)
    graph.fit([x_train, x_train], y_train,
              epochs=1,
              batch_size=100,
              validation_data=([x_train, x_train], y_train),
              validation_split=0.5,
              verbose=False)


def test_input_output_disconnect(tmp_dir):
    input_node1 = ak.Input()
    output_node = input_node1
    _ = ak.DenseBlock()(output_node)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    with pytest.raises(ValueError) as info:
        ak.GraphAutoModel(input_node1, output_node, directory=tmp_dir)
    assert str(info.value) == 'Inputs and outputs not connected.'


def test_hyper_graph_cycle(tmp_dir):
    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    head = ak.RegressionHead()
    output_node = head(output_node)
    head.outputs = output_node1

    with pytest.raises(ValueError) as info:
        ak.GraphAutoModel([input_node1, input_node2],
                          output_node,
                          directory=tmp_dir)
    assert str(info.value) == 'The network has a cycle.'


def test_input_missing(tmp_dir):
    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    output_node = ak.RegressionHead()(output_node)

    with pytest.raises(ValueError) as info:
        ak.GraphAutoModel(input_node1, output_node, directory=tmp_dir)
    assert str(info.value).startswith('A required input is missing for HyperModel')


@mock.patch('kerastuner.engine.tuner.Tuner.search',
            side_effect=lambda *args, **kwargs: None)
def test_auto_model_basic(_, tmp_dir):
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_dir,
                              max_trials=2)
    auto_model.fit(x_train, y_train, epochs=2, validation_split=0.2)


def test_structured_data_input(tmp_dir):
    num_data = 500
    data = common.structured_data(num_data)
    x_train = data
    y = np.random.randint(0, 3, num_data)
    y_train = y

    input_node = ak.StructuredDataInput(
        column_names=common.COLUMN_NAMES_FROM_NUMPY,
        column_types=common.COLUMN_TYPES_FROM_NUMPY)
    output_node = input_node
    output_node = ak.StructuredDataBlock()(output_node)
    output_node = ak.ClassificationHead(loss='categorical_crossentropy',
                                        metrics=['accuracy'])(output_node)

    auto_model = ak.GraphAutoModel(input_node,
                                   output_node,
                                   directory=tmp_dir,
                                   max_trials=1)
    auto_model.fit(x_train, y_train, epochs=1,
                   validation_data=(x_train, y_train))
    auto_model.predict(x_train)
