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
    # generate high_level dataset
    data_num = 500
    feature_num = 8
    nan_num = 100
    data = []
    # 12 classes
    career = ['doctor', 'nurse', 'driver', 'chef', 'teacher', 'writer',
              'actress', 'engineer', 'lawyer', 'realtor', 'agent', 'pilot']
    # 15 classes
    states = ['CA', 'FL', 'GA', 'IL', 'MD',
              'MA', 'MI', 'MN', 'NJ', 'NY',
              'NC', 'PA', 'TX', 'UT', 'VA']
    # 13 classes
    years = ['first', 'second', 'third', 'fourth', 'fifth',
             'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
             'eleventh', 'twelfth', 'thirteenth']
    # 10 classes
    color = ['red', 'orange', 'yellow', 'green', 'blue',
             'purple', 'beige', 'pink', 'silver', 'gold']
    # 3 classes
    size = ['S', 'M', 'L']
    boolean = ['True', 'False']
    career_states = []  # 180 classes
    career_years = []  # 156 classes
    career_color = []  # 120 classes
    career_size = []  # 36 classes
    for c in career:
        for s in states:
            career_states.append(c+'_'+s)
        for y in years:
            career_years.append(c+'_'+y)
        for r in color:
            career_color.append(c+'_'+r)
        for g in size:
            career_size.append(c+'_'+g)

    col_bool = np.random.choice(boolean, data_num).reshape(data_num, 1)
    col_num_to_cat = np.random.randint(20, 41, size=data_num).reshape(data_num, 1)
    col_float = 100*np.random.random(data_num,).reshape(data_num, 1)
    col_int = np.random.randint(2000, 4000, size=data_num).reshape(data_num, 1)
    col_morethan_32 = np.random.choice(career_size, data_num).reshape(data_num, 1)
    col1_morethan_100 = np.random.choice(career_states,
                                         data_num).reshape(data_num, 1)
    col2_morethan_100 = np.random.choice(career_years,
                                         data_num).reshape(data_num, 1)
    col3_morethan_100 = np.random.choice(career_color,
                                         data_num).reshape(data_num, 1)
    data = np.concatenate((col_bool, col_num_to_cat, col_float, col_int,
                           col_morethan_32, col1_morethan_100, col2_morethan_100,
                           col3_morethan_100), axis=1)
    # generate np.nan data
    for i in range(nan_num):
        row = np.random.randint(0, data_num)
        col = np.random.randint(0, feature_num)
        data[row][col] = np.nan

    dataset = tf.data.Dataset.from_tensor_slices(data)
    assembler = meta_model.StructuredDataAssembler()
    for line in dataset:
        assembler.update(line)

    assembler.assemble(node.Input(),task='classification')
