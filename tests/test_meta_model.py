import numpy as np
import tensorflow as tf

from autokeras import meta_model
from autokeras.hypermodel import node


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
    data = np.array([
        [1, True, 'ab'],
        [2, False, 'cd'],
        [3, True, 'ef'],
        [np.nan, True, 'gh'],
        [5, False, np.nan],
        [6, True, 'ab'],
    ])
    data = np.array([
        [1, True, 'ab', 1.1],
        [1, False, 'cd', 2.2],
        [1, True, 'ef', 2.1],
        [1, True, 'gh', 'a'],
        [1, False, np.nan, 3.3],
        [1, True, 'ab', 3.4],
        [1, True, 'ab', 1.1],
        [1, False, 'cd', 2.2],
        [1, True, 'ef', 2.1],
        [1, True, 'gh', 'a'],
        [1, False, np.nan, 3.3],
        [1, True, 'ab', 3.4],
        [1, True, 'ab', 1.1],
        [1, False, 'cd', 2.2],
        [1, True, 'ef', 2.1],
        [1, True, 'gh', 'a'],
        [1, False, np.nan, 3.3],
        [1, True, 'ab', 3.4],
        [1, True, 'ab', 1.1],
        [1, False, 'cd', 2.2],
        [1, True, 'ef', 2.1],
        [1, True, 'gh', 'a'],
        [1, False, np.nan, 3.3],
        [1, True, 'ab', 3.4],
    ])
    dataset = tf.data.Dataset.from_tensor_slices(data)
    assembler = meta_model.StructuredDataAssembler()
    for line in dataset:
        assembler.update(line)

    assembler.assemble(node.Input())
