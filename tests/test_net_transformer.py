import torch

from autokeras.nn.generator import CnnGenerator
from autokeras.nn.graph import Graph
from autokeras.net_transformer import *
from tests.common import get_conv_dense_model, get_pooling_model, get_conv_data


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
