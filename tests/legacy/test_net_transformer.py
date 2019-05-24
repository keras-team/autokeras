from autokeras.legacy.constant import Constant
from autokeras.legacy.net_transformer import to_wider_graph, to_deeper_graph, to_skip_connection_graph, transform
from autokeras.legacy.nn.graph import Graph
from tests.legacy.common import get_conv_dense_model, get_pooling_model


def test_wider():
    model = to_wider_graph(get_pooling_model())
    assert isinstance(model, Graph)


def test_wider_dense():
    model = to_wider_graph(get_pooling_model())
    assert isinstance(model, Graph)


def test_deeper():
    model = to_deeper_graph(get_conv_dense_model())
    assert isinstance(model, Graph)


def test_skip():
    model = to_skip_connection_graph(get_pooling_model())
    assert isinstance(model, Graph)


def test_transform():
    models = transform(get_pooling_model())
    assert len(models) == Constant.N_NEIGHBOURS
