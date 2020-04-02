import kerastuner
import tensorflow as tf

import autokeras as ak
from autokeras import adapters
from autokeras import graph as graph_module
from autokeras.hypermodels import wrapper
from tests import utils


def test_image_block():
    block = wrapper.ImageBlock(normalize=None, augment=None)
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, ak.ImageInput(shape=(32, 32, 3)).build())

    assert utils.name_in_hps('block_type', hp)
    assert utils.name_in_hps('normalize', hp)
    assert utils.name_in_hps('augment', hp)


def test_text_block():
    block = wrapper.TextBlock()
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, ak.TextInput(shape=(1,)).build())

    assert utils.name_in_hps('vectorizer', hp)


def test_structured_data_block():
    block = wrapper.StructuredDataBlock()
    block.column_names = ['0', '1']
    block.column_types = {
        '0': adapters.CATEGORICAL,
        '1': adapters.CATEGORICAL,
    }
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.column_names = ['0', '1']
    block.column_types = {
        '0': adapters.CATEGORICAL,
        '1': adapters.CATEGORICAL,
    }
    output = block.build(hp, ak.StructuredDataInput(shape=(2,)).build())

    assert isinstance(output, tf.Tensor)


def test_timeseries_block():
    block = wrapper.TimeseriesBlock()
    hp = kerastuner.HyperParameters()
    block.column_names = ['0', '1']
    block.column_types = {
        '0': adapters.NUMERICAL,
        '1': adapters.NUMERICAL,
    }
    block = graph_module.deserialize(graph_module.serialize(block))

    block.column_names = ['0', '1']
    block.column_types = {
        '0': adapters.NUMERICAL,
        '1': adapters.NUMERICAL,
    }
    output = block.build(hp, ak.TimeseriesInput(shape=(32,), lookback=2).build())

    assert isinstance(output, tf.Tensor)
