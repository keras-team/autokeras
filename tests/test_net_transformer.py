from unittest.mock import patch
from autokeras.net_transformer import *
from tests.common import get_conv_dense_model


@patch('autokeras.net_transformer.Graph.to_wider_model', side_effect=lambda a, b: None)
def test_wider(_):
    model = to_wider_model(get_conv_dense_model())
    assert model is None


@patch('autokeras.net_transformer.Graph.to_conv_deeper_model', side_effect=lambda a, b: None)
@patch('autokeras.net_transformer.Graph.to_dense_deeper_model', side_effect=lambda a: None)
def test_deeper(_, _1):
    model = to_deeper_model(get_conv_dense_model())
    assert model is None


@patch('autokeras.net_transformer.Graph.to_add_skip_model', side_effect=lambda a, b: None)
@patch('autokeras.net_transformer.Graph.to_concat_skip_model', side_effect=lambda a, b: None)
def test_skip(_, _1):
    model = to_skip_connection_model(get_conv_dense_model())
    assert model is None


@patch('autokeras.net_transformer.Graph.to_wider_model', side_effect=lambda a, b: None)
@patch('autokeras.net_transformer.Graph.to_conv_deeper_model', side_effect=lambda a, b: None)
@patch('autokeras.net_transformer.Graph.to_dense_deeper_model', side_effect=lambda a: None)
@patch('autokeras.net_transformer.Graph.to_add_skip_model', side_effect=lambda a, b: None)
@patch('autokeras.net_transformer.Graph.to_concat_skip_model', side_effect=lambda a, b: None)
def test_transform(_, _1, _2, _3, _4):
    models = transform(get_conv_dense_model())
    assert len(models) == constant.N_NEIGHBORS
