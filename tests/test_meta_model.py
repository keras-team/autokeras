import tensorflow as tf

from autokeras import meta_model
from autokeras.hypermodel import node
from .common import structured_data


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
    data = structured_data()
    dataset = tf.data.Dataset.from_tensor_slices(data)
    assembler = meta_model.StructuredDataAssembler(
        column_names=[
                        'bool_',
                        'num_to_cat_',
                        'float_',
                        'int_',
                        'morethan_32_',
                        'col1_morethan_100_',
                        'col2_morethan_100_',
                        'col3_morethan_100_'])
    for line in dataset:
        assembler.update(line)

    assembler.assemble(node.Input())
