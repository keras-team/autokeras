import kerastuner

import autokeras as ak
from autokeras.hypermodel import hyperblock as hyperblock_module
from tests import common


def test_image_block():
    block = hyperblock_module.ImageBlock(normalize=None, augment=None)
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input())

    assert common.name_in_hps('block_type', hp)
    assert common.name_in_hps('normalize', hp)
    assert common.name_in_hps('augment', hp)


def test_text_block():
    block = hyperblock_module.TextBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.TextInput())

    assert common.name_in_hps('vectorizer', hp)


def test_structured_data_block():
    block = hyperblock_module.StructuredDataBlock()
    block.num_heads = 1
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input())

    assert common.name_in_hps('module_type', hp)
