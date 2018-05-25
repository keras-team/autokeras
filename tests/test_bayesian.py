from autokeras.bayesian import *
from autokeras.graph import Graph
from tests.common import get_add_skip_model, get_concat_skip_model


def test_edit_distance():
    descriptor1 = Graph(get_add_skip_model()).extract_descriptor()
    descriptor2 = Graph(get_concat_skip_model()).extract_descriptor()
    assert edit_distance(descriptor1, descriptor2, 1.0) == 2.0


def test_gpr():
    gpr = IncrementalGaussianProcess(1.0)
    gpr.first_fit([Graph(get_add_skip_model()).extract_descriptor()], [0.5])
    assert gpr.first_fitted

    gpr.incremental_fit([Graph(get_concat_skip_model()).extract_descriptor()], [0.6])
    assert abs(gpr.predict(np.array([Graph(get_concat_skip_model()).extract_descriptor()]))[0] - 0.6) < 1e-4
