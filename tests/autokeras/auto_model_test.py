from unittest import mock

import numpy as np
import pytest

import autokeras as ak


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


@mock.patch('autokeras.tuner.RandomSearch')
@mock.patch('autokeras.hypermodel.graph.GraphHyperModel')
def test_evaluate(graph, tuner, tmp_dir):
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100, 1)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    auto_model = ak.GraphAutoModel(input_node,
                                   output_node,
                                   directory=tmp_dir,
                                   max_trials=1)
    auto_model.fit(x_train, y_train, epochs=1, validation_data=(x_train, y_train))
    auto_model.evaluate(x_train, y_train)
    assert tuner.called
    assert graph.called


@mock.patch('autokeras.tuner.RandomSearch')
@mock.patch('autokeras.hypermodel.graph.GraphHyperModel')
def test_auto_model_predict(graph, tuner, tmp_dir):
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_dir,
                              max_trials=2)
    auto_model.fit(x_train, y_train, epochs=2, validation_split=0.2)
    auto_model.predict(x_train)
    assert tuner.called
    assert graph.called
