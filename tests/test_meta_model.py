import tensorflow as tf

from autokeras import meta_model


def test_text_assembler():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together aa.']
    assembler = meta_model.TextAssembler()
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    for x in dataset:
        assembler.update(x)
    assert assembler.sw_ratio() == 0.5
