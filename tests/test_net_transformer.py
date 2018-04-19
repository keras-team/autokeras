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
    assert len(models) == 12
