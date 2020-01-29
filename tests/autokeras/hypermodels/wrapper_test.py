import kerastuner
import tensorflow as tf

import autokeras as ak
from autokeras.hypermodels import wrapper
from autokeras.adapters import input_adapter
from tests import utils


def test_image_block():
    block = wrapper.ImageBlock(normalize=None, augment=None)
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.ImageInput(shape=(32, 32, 3)).build())

    assert utils.name_in_hps('block_type', hp)
    assert utils.name_in_hps('normalize', hp)
    assert utils.name_in_hps('augment', hp)


def test_text_block():
    block = wrapper.TextBlock()
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.TextInput(shape=(1,)).build())

    assert utils.name_in_hps('vectorizer', hp)


def test_structured_data_block():
    block = wrapper.StructuredDataBlock()
    block.num_heads = 1
    block.column_names = ['0', '1']
    block.column_types = {
        '0': input_adapter.StructuredDataInputAdapter.CATEGORICAL,
        '1': input_adapter.StructuredDataInputAdapter.CATEGORICAL,
    }
    hp = kerastuner.HyperParameters()

    output = block.build(hp, ak.StructuredDataInput(shape=(2,)).build())

    assert isinstance(output, tf.Tensor)
