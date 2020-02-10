import kerastuner

import autokeras as ak
from autokeras import graph as graph_module
from autokeras.hypermodels import reduction
from tests import utils


def test_merge():
    input_shape_1 = (32,)
    input_shape_2 = (4, 8)
    block = reduction.Merge()
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, [ak.Input(shape=input_shape_1).build(),
                     ak.Input(shape=input_shape_2).build()])

    assert utils.name_in_hps('merge_type', hp)


def test_temporal_reduction():
    input_shape = (32, 10)
    block = reduction.TemporalReduction()
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, ak.Input(shape=input_shape).build())

    assert utils.name_in_hps('reduction_type', hp)


def test_spatial_reduction():
    input_shape = (32, 32, 3)
    block = reduction.SpatialReduction()
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, ak.Input(shape=input_shape).build())

    assert utils.name_in_hps('reduction_type', hp)
