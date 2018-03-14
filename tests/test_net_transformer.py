from unittest.mock import patch

from autokeras.graph import Graph
from autokeras.net_transformer import *
from autokeras.stub import to_stub_model
from tests.common import get_conv_dense_model


def test_wider():
    model = to_wider_graph(Graph(to_stub_model(get_conv_dense_model())))
    assert isinstance(model, Graph)


def test_deeper():
    model = to_deeper_graph(Graph(to_stub_model(get_conv_dense_model())))
    assert isinstance(model, Graph)


def test_skip():
    model = to_skip_connection_graph(Graph(to_stub_model(get_conv_dense_model())))
    assert model is None


def test_transform():
    models = transform(Graph(get_conv_dense_model()))
    assert len(models) == constant.N_NEIGHBORS
