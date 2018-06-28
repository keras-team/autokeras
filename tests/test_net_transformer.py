from autokeras.generator import DefaultClassifierGenerator
from autokeras.graph import Graph
from autokeras.net_transformer import *
from autokeras.stub import to_stub_model
from tests.common import get_conv_dense_model, get_pooling_model


def test_wider():
    model = to_wider_graph(Graph(get_pooling_model(), False))
    assert isinstance(model, Graph)


def test_wider_dense():
    model = to_wider_graph(Graph(get_pooling_model(), False))
    assert isinstance(model, Graph)


def test_deeper():
    model = to_deeper_graph(Graph(get_conv_dense_model(), False))
    assert isinstance(model, Graph)


def test_skip():
    model = to_skip_connection_graph(Graph(get_pooling_model(), False))
    assert isinstance(model, Graph)


def test_transform():
    models = transform(Graph(get_pooling_model(), False))
    assert len(models) == constant.N_NEIGHBOURS


def test_legal_graph():
    graph = Graph(get_pooling_model(), False)
    graph.to_add_skip_model(1, 5)
    assert legal_graph(graph)
    graph.to_add_skip_model(1, 5)
    assert not legal_graph(graph)


def test_legal_graph2():
    graph = Graph(get_pooling_model(), False)
    graph.to_concat_skip_model(1, 5)
    assert legal_graph(graph)
    graph.to_concat_skip_model(1, 5)
    assert not legal_graph(graph)


def test_default_transform():
    graphs = default_transform(DefaultClassifierGenerator(10, (28, 28, 1)).generate())
    # print()
    # for index, layer in enumerate(graphs[0].layer_list):
    #     print(index, layer)
    graphs[0].produce_model()
    assert len(graphs) == 1
    assert len(graphs[0].layer_list) == 41
