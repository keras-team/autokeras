from autokeras.bayesian import *
from autokeras.nn.layers import StubConv
from tests.common import get_add_skip_model, get_concat_skip_model, get_conv_dense_model


def test_layer_distance():
    layer1 = StubConv(5, 5, 3, 2)
    layer2 = StubConv(5, 1, 1, 1)
    assert layer_distance(layer1, layer2) == 5.9 / 9


def test_edit_distance():
    descriptor1 = get_add_skip_model().extract_descriptor()
    descriptor2 = get_concat_skip_model().extract_descriptor()
    assert edit_distance(descriptor1, descriptor2) == 12.0


def test_edit_distance2():
    descriptor1 = get_conv_dense_model().extract_descriptor()
    graph = get_conv_dense_model()
    graph.to_wider_model(4, 6)
    graph.to_wider_model(9, 3)
    descriptor2 = graph.extract_descriptor()
    assert edit_distance(descriptor1, descriptor2) == 2.0 / 9


def test_bourgain_embedding():
    assert bourgain_embedding_matrix([[0]]).shape == (1, 1)
    assert bourgain_embedding_matrix([[1, 0], [0, 1]]).shape == (2, 2)


def test_gpr():
    gpr = IncrementalGaussianProcess()
    gpr.first_fit([get_add_skip_model().extract_descriptor()], [0.5])
    assert gpr.first_fitted

    gpr.incremental_fit([get_concat_skip_model().extract_descriptor()], [0.6])
    assert abs(gpr.predict(np.array([get_add_skip_model().extract_descriptor()]))[0] - 0.5) < 1e-4
    assert abs(gpr.predict(np.array([get_concat_skip_model().extract_descriptor()]))[0] - 0.6) < 1e-4


def test_elem_queue():
    elem1 = Elem(1, 2, 3)
    elem2 = Elem(2, 3, 4)
    pq = PriorityQueue()
    pq.put(elem1)
    pq.put(elem2)
    assert pq.get() == elem1
    assert pq.get() == elem2

    elem1 = ReverseElem(1, 2, 3)
    elem2 = ReverseElem(2, 3, 4)
    pq = PriorityQueue()
    pq.put(elem1)
    pq.put(elem2)
    assert pq.get() == elem2
    assert pq.get() == elem1


def test_search_tree():
    tree = SearchTree()
    tree.add_child(-1, 0)
    tree.add_child(0, 1)
    tree.add_child(0, 2)
    assert len(tree.adj_list) == 3
