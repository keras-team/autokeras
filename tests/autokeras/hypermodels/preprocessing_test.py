import kerastuner

import autokeras as ak
from autokeras import graph as graph_module
from autokeras.hypermodels import preprocessing
from tests import utils


def test_imag_augmentation():
    input_shape = (32, 32, 3)
    block = preprocessing.ImageAugmentation()
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, ak.Input(shape=input_shape).build())

    assert utils.name_in_hps('vertical_flip', hp)
    assert utils.name_in_hps('horizontal_flip', hp)
