import kerastuner
import tensorflow as tf

import autokeras as ak
from autokeras.hypermodels import wrapper
from tests import utils


def test_image_block():
    block = wrapper.ImageBlock(normalize=None, augment=None)
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.ImageInput(shape=(32, 32, 3)).build())

    assert utils.name_in_hps('block_type', hp)
    assert utils.name_in_hps('normalize', hp)
    assert utils.name_in_hps('augment', hp)


def test_text_block():
    block = wrapper.TextBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.TextInput(shape=(1,)).build())

    assert utils.name_in_hps('vectorizer', hp)


def test_structured_data_block():
    block = wrapper.StructuredDataBlock()
    block.num_heads = 1
    block.column_names = ['0', '1']
    block.column_types = {
        '0': ak.StructuredDataInput.CATEGORICAL,
        '1': ak.StructuredDataInput.CATEGORICAL,
    }
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    output = block.build(hp, ak.StructuredDataInput(shape=(2,)).build())

    assert isinstance(output, tf.Tensor)
