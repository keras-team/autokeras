import tensorflow as tf

import autokeras as ak
from autokeras import meta_model
from autokeras.hypermodel import node
from tests import common


def test_text_assembler():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together aa.']
    assembler = meta_model.TextAssembler()
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    for x in dataset:
        assembler.update(x)
    assert assembler.sw_ratio() == 0.5


def test_structured_data_assembler():
    data = common.generate_structured_data()
    dataset = tf.data.Dataset.from_tensor_slices(data)
    assembler = meta_model.StructuredDataAssembler()
    for line in dataset:
        assembler.update(line)

    input_node = node.StructuredDataInput()
    assembler.assemble(input_node)
    assert isinstance(input_node.out_blocks[0], ak.StructuredDataBlock)
