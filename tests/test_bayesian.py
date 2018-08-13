from autokeras.bayesian import *
from tests.common import get_add_skip_model, get_concat_skip_model, get_conv_dense_model


def test_edit_distance():
    descriptor1 = get_add_skip_model().extract_descriptor()
    descriptor2 = get_concat_skip_model().extract_descriptor()
    assert edit_distance(descriptor1, descriptor2, 1.0) == 2.0


def test_edit_distance2():
    descriptor1 = get_conv_dense_model().extract_descriptor()
    graph = get_conv_dense_model()
    graph.to_conv_deeper_model(1, 3)
    graph.to_wider_model(5, 6)
    graph.to_wider_model(17, 3)
    descriptor2 = graph.extract_descriptor()
    assert edit_distance(descriptor1, descriptor2, 1.0) == 1.5


def test_bourgain_embedding():
    assert bourgain_embedding_matrix([[0]]).shape == (1, 1)
    assert bourgain_embedding_matrix([[1, 0], [0, 1]]).shape == (2, 2)


def test_gpr():
    gpr = IncrementalGaussianProcess(1.0)
    gpr.first_fit([get_add_skip_model().extract_descriptor()], [0.5])
    assert gpr.first_fitted

    gpr.incremental_fit([get_concat_skip_model().extract_descriptor()], [0.6])
    assert abs(gpr.predict(np.array([get_add_skip_model().extract_descriptor()]))[0] - 0.5) < 1e-4
    assert abs(gpr.predict(np.array([get_concat_skip_model().extract_descriptor()]))[0] - 0.6) < 1e-4
