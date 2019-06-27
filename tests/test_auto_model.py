import pytest
import numpy as np
import kerastuner
import autokeras as ak


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


def test_graph_auto_model_basic(tmp_dir):
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    input_node.shape = (32,)
    output_node[0].shape = (1,)

    graph = ak.GraphAutoModel(input_node, output_node, directory=tmp_dir)
    model = graph.build(kerastuner.HyperParameters())
    model.fit(x_train, y_train, epochs=1, batch_size=100, verbose=False)
    result = model.predict(x_train)

    assert result.shape == (100, 1)


def test_merge(tmp_dir):
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100)

    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    output_node = ak.RegressionHead()(output_node)

    input_node1.shape = (32,)
    input_node2.shape = (32,)
    output_node[0].shape = (1,)

    graph = ak.GraphAutoModel([input_node1, input_node2],
                              output_node,
                              directory=tmp_dir)
    model = graph.build(kerastuner.HyperParameters())
    model.fit([x_train, x_train], y_train, epochs=1, batch_size=100, verbose=False)
    result = model.predict([x_train, x_train])

    assert result.shape == (100, 1)


def test_input_output_disconnect(tmp_dir):
    input_node1 = ak.Input()
    output_node = input_node1
    _ = ak.DenseBlock()(output_node)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    input_node.shape = (32,)
    output_node[0].shape = (1,)

    with pytest.raises(ValueError) as info:
        graph = ak.GraphAutoModel(input_node1,
                                  output_node,
                                  directory=tmp_dir)
        graph.build(kerastuner.HyperParameters())
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

    input_node1.shape = (32,)
    input_node2.shape = (32,)
    output_node[0].shape = (1,)

    with pytest.raises(ValueError) as info:
        graph = ak.GraphAutoModel([input_node1, input_node2],
                                  output_node,
                                  directory=tmp_dir)
        graph.build(kerastuner.HyperParameters())
    assert str(info.value) == 'The network has a cycle.'


def test_input_missing(tmp_dir):
    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    output_node = ak.RegressionHead()(output_node)

    input_node1.shape = (32,)
    input_node2.shape = (32,)
    output_node[0].shape = (1,)

    with pytest.raises(ValueError) as info:
        graph = ak.GraphAutoModel(input_node1, output_node, directory=tmp_dir)
        graph.build(kerastuner.HyperParameters())
    assert str(info.value).startswith('A required input is missing for HyperModel')


def test_auto_model_basic(tmp_dir):
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100)

    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_dir,
                              trials=2)
    auto_model.fit(x_train, y_train, epochs=2)
    result = auto_model.predict(x_train)

    assert result.shape == (100, 1)
