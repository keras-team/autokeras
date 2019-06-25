import pytest

from autokeras.auto.auto_model import *
from autokeras import const
from autokeras.hypermodel.hyper_block import DenseBlock, Merge
from autokeras.hypermodel.hyper_head import RegressionHead
from autokeras.hypermodel.hyper_node import Input


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


def test_hyper_graph_basic(tmp_dir):
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100)

    input_node = Input()
    output_node = input_node
    output_node = DenseBlock()(output_node)
    output_node = RegressionHead()(output_node)

    input_node.shape = (32,)
    output_node[0].shape = (1,)

    graph = GraphAutoModel(input_node, output_node, directory=tmp_dir)
    model = graph.build(kerastuner.HyperParameters())
    model.fit(x_train, y_train, epochs=1, batch_size=100, verbose=False)
    result = model.predict(x_train)

    assert result.shape == (100, 1)


def test_merge(tmp_dir):
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100)

    input_node1 = Input()
    input_node2 = Input()
    output_node1 = DenseBlock()(input_node1)
    output_node2 = DenseBlock()(input_node2)
    output_node = Merge()([output_node1, output_node2])
    output_node = RegressionHead()(output_node)

    input_node1.shape = (32,)
    input_node2.shape = (32,)
    output_node[0].shape = (1,)

    graph = GraphAutoModel([input_node1, input_node2],
                           output_node,
                           directory=tmp_dir)
    model = graph.build(kerastuner.HyperParameters())
    model.fit([x_train, x_train], y_train, epochs=1, batch_size=100, verbose=False)
    result = model.predict([x_train, x_train])

    assert result.shape == (100, 1)


def test_input_output_disconnect(tmp_dir):
    input_node1 = Input()
    output_node = input_node1
    _ = DenseBlock()(output_node)

    input_node = Input()
    output_node = input_node
    output_node = DenseBlock()(output_node)
    output_node = RegressionHead()(output_node)

    input_node.shape = (32,)
    output_node[0].shape = (1,)

    with pytest.raises(ValueError) as info:
        graph = GraphAutoModel(input_node1,
                               output_node,
                               directory=tmp_dir)
        graph.build(kerastuner.HyperParameters())
    assert str(info.value) == 'Inputs and outputs not connected.'


def test_hyper_graph_cycle(tmp_dir):
    input_node1 = Input()
    input_node2 = Input()
    output_node1 = DenseBlock()(input_node1)
    output_node2 = DenseBlock()(input_node2)
    output_node = Merge()([output_node1, output_node2])
    head = RegressionHead()
    output_node = head(output_node)
    head.outputs = output_node1

    input_node1.shape = (32,)
    input_node2.shape = (32,)
    output_node[0].shape = (1,)

    with pytest.raises(ValueError) as info:
        graph = GraphAutoModel([input_node1, input_node2],
                               output_node,
                               directory=tmp_dir)
        graph.build(kerastuner.HyperParameters())
    assert str(info.value) == 'The network has a cycle.'


def test_input_missing(tmp_dir):
    input_node1 = Input()
    input_node2 = Input()
    output_node1 = DenseBlock()(input_node1)
    output_node2 = DenseBlock()(input_node2)
    output_node = Merge()([output_node1, output_node2])
    output_node = RegressionHead()(output_node)

    input_node1.shape = (32,)
    input_node2.shape = (32,)
    output_node[0].shape = (1,)

    with pytest.raises(ValueError) as info:
        graph = GraphAutoModel(input_node1, output_node, directory=tmp_dir)
        graph.build(kerastuner.HyperParameters())
    assert str(info.value).startswith('A required input is missing for HyperModel')


def test_auto_model_basic(tmp_dir):
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100)

    input_node = Input()
    output_node = input_node
    output_node = DenseBlock()(output_node)
    output_node = RegressionHead()(output_node)

    auto_model = GraphAutoModel(input_node, output_node, directory=tmp_dir)
    const.Constant.NUM_TRAILS = 2
    auto_model.fit(x_train, y_train, epochs=2)
    result = auto_model.predict(x_train)

    assert result.shape == (100, 1)
