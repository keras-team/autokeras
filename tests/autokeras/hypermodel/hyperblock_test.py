import kerastuner

import autokeras as ak
from autokeras.hypermodel import hyperblock as hyperblock_module
from tests import common


def test_image_block():
    block = hyperblock_module.ImageBlock(normalize=None, augment=None)
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input())

    if not (common.name_in_hps('block_type', hp) and
            common.name_in_hps('normalize', hp) and
            common.name_in_hps('augment', hp)):
        raise AssertionError()


def test_text_block():
    block = hyperblock_module.TextBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.TextInput())

    if not common.name_in_hps('vectorizer', hp):
        raise AssertionError()


def test_structured_data_block():
    block = hyperblock_module.StructuredDataBlock()
    block.heads = [ak.ClassificationHead()]
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input())

    if not common.name_in_hps('module_type', hp):
        raise AssertionError()
