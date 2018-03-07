from autokeras.bayesian import *
from autokeras.graph import Graph
from tests.common import get_add_skip_model, get_concat_skip_model


def test_edit_distance():
    descriptor1 = Graph(get_add_skip_model()).extract_descriptor()
    descriptor2 = Graph(get_concat_skip_model()).extract_descriptor()
    assert edit_distance(descriptor1, descriptor2) == 2
