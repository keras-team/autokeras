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
        ['doctor', 'orange', 1, True, 'ab', 1.1, 10],
        ['nurse', 'pink', 1, False, 'cd', 2.2, 13],
        ['driver', 'silver', 1, True, 'ef', 2.1, 76],
        ['chef', 'gold', 1, True, 'gh', np.nan, 98],
        ['teacher', 'blue', 1, False, np.nan, 3.3, 65],
        ['writer', 'green', 0, True, 'ab', 3.4, 42],
        ['actress', 'red', 1, True, 'ab', 1.1, 21],
        ['engineer', 'purple', 1, False, 'cd', 2.2, 77],
        ['lawyer', 'grey', 1, True, 'ef', 2.1, 29], 
        ['realtor', 'maroon', 1, True, 'gh', 4.6, 61],

        ['agent', 'coral', 1, False, np.nan, 3.3, 32],
        ['pilot', 'beige', 1, True, 'ab', 3.4, 75],
        ['realtor', 'grey', 1, True, 'ab', 1.1, 39],
        ['writer', 'silver', 1, False, 'cd', 2.2, 34],
        ['teacher', 'red', 1, np.nan, 'ef', 2.1, 101],
        ['writer', 'maroon', 1, True, 'gh', 12.3, 231],
        ['chef', 'grey', 0, False, np.nan, 3.3, 91],
        ['writer', 'silver', 1, True, 'ab', 3.4, 201],
        ['teacher', 'maroon', 1, True, 'ab', 1.1, 300],
        ['engineer', 'grey', 1, False, 'cd', 2.2, 67],

        ['chef', 'green', 1, True, 'ef', 2.1, 201],
        ['nurse', 'pink', 1, True, 'gh', 7.1, 176],
        ['doctor', 'coral', 1, False, np.nan, 3.3, 192],
        ['realtor', 'orange', 0, True, 'ab', 3.4, 172],
        ['actress', 'red', 1, True, 'ab', 1.1, 19],
        ['pilot', 'coral', 1, False, 'cd', 2.2, 102],
        ['lawyer', 'gold', 1, True, 'ef', 2.1, 241],
        ['doctor', 'beige', 1, True, 'gh', 5.9, 114],
        ['engineer', 'pink', 1, False, np.nan, 3.3, 99],
        ['lawyer', 'pink', 1, True, 'ab', 3.4, 70],

        ['agent', 'beige', 1, True, 'ab', 1.1, 102],
        ['engineer', 'yellow', 1, False, 'cd', 2.2, 70],
        ['lawyer', 'maroon', 1, True, 'ef', 2.1, 231],
        ['realtor', 'silver', 0, True, 'gh', 4.34, 47],
        ['chef', 'gold', 1, False, np.nan, 3.3, 231],
        ['driver', 'orange', 1, True, 'ab', 3.4, 96],
        ['nurse', 'orange', 1, True, 'ab', 1.1, 155],
        ['pilot', 'yellow', 0, False, 'cd', 4.87, 102],
        ['doctor', 'green', 1, np.nan, 'ef', 2.1, 46],
        ['teacher', 'blue', 1, True, 'gh', 8.12, 24],

        ['sailor', 'green', 1, False, np.nan, 3.3, np.nan],
        ['actress', 'gold', 1, True, 'ab', 3.4, 77],
        ['sailor', 'red', 1, True, 'ab', 1.1, np.nan],
        ['agent', 'blue', 1, False, 'cd', 2.2, 77],
        ['driver', 'coral', 1, True, 'ef', 2.1, 233],
        ['sailor', 'pink', 1, True, 'gh', 3.89, 119],
        ['actress', 'beige', 1, False, np.nan, 3.3, 102],
        ['nurse', 'orange', 1, True, 'ab', 3.4, 231],
        ['pilot', 'blue', 1, False, np.nan, 3.3, 120],
        ['agent', 'green', 0, True, 'ab', 3.4, np.nan],
    ])
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

    assembler.assemble(node.Input())
test_structured_data_assembler()