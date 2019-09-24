import numpy as np

COLUMN_NAMES_FROM_NUMPY = ['bool_',
                           'num_to_cat_',
                           'float_',
                           'int_',
                           'morethan_32_',
                           'col1_morethan_100_',
                           'col2_morethan_100_',
                           'col3_morethan_100_']
COLUMN_TYPES_FROM_NUMPY = {
                            'bool_': 'categorical',
                            'num_to_cat_': 'categorical',
                            'float_': 'numerical',
                            'int_': 'numerical',
                            'morethan_32_': 'categorical',
                            'col1_morethan_100_': 'categorical',
                            'col2_morethan_100_': 'categorical',
                            'col3_morethan_100_': 'categorical'}
COLUMN_NAMES_FROM_CSV = [
                         'sex_',
                         'age_',
                         'n_siblings_spouses_',
                         'parch_',
                         'fare_',
                         'class_',
                         'deck_',
                         'embark_town_',
                         'alone_']
LESS_COLUMN_NAMES_FROM_CSV = [
                         'age_',
                         'n_siblings_spouses_',
                         'parch_',
                         'fare_',
                         'class_',
                         'deck_',
                         'embark_town_',
                         'alone_']
COLUMN_TYPES_FROM_CSV = {
                          'sex_': 'categorical',
                          'age_': 'numerical',
                          'n_siblings_spouses_': 'categorical',
                          'parch_': 'categorical',
                          'fare_': 'numerical',
                          'class_': 'categorical',
                          'deck_': 'categorical',
                          'embark_town_': 'categorical',
                          'alone_': 'categorical'}
FALSE_COLUMN_TYPES_FROM_CSV = {
                          'sex_': 'cat',
                          'age_': 'num',
                          'n_siblings_spouses_': 'cat',
                          'parch_': 'categorical',
                          'fare_': 'numerical',
                          'class_': 'categorical',
                          'deck_': 'categorical',
                          'embark_town_': 'categorical',
                          'alone_': 'categorical'}
PARTIAL_COLUMN_TYPES_FROM_CSV = {
                          'fare': 'numerical',
                          'class': 'categorical',
                          'deck': 'categorical',
                          'embark_town': 'categorical',
                          'alone': 'categorical'}
TRAIN_FILE_PATH = r'tests/resources/titanic/train.csv'
TEST_FILE_PATH = r'tests/resources/titanic/eval.csv'


def structured_data(num_data=500):
    # generate high_level dataset
    num_feature = 8
    num_nan = 100
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

    np.random.seed(0)
    col_bool = np.random.choice(boolean, num_data).reshape(num_data, 1)
    col_num_to_cat = np.random.randint(20, 41, size=num_data).reshape(num_data, 1)
    col_float = 100*np.random.random(num_data,).reshape(num_data, 1)
    col_int = np.random.randint(2000, 4000, size=num_data).reshape(num_data, 1)
    col_morethan_32 = np.random.choice(career_size, num_data).reshape(num_data, 1)
    col1_morethan_100 = np.random.choice(career_states,
                                         num_data).reshape(num_data, 1)
    col2_morethan_100 = np.random.choice(career_years,
                                         num_data).reshape(num_data, 1)
    col3_morethan_100 = np.random.choice(career_color,
                                         num_data).reshape(num_data, 1)
    data = np.concatenate((col_bool, col_num_to_cat, col_float, col_int,
                           col_morethan_32, col1_morethan_100, col2_morethan_100,
                           col3_morethan_100), axis=1)
    # generate np.nan data
    for i in range(num_nan):
        row = np.random.randint(0, num_data)
        col = np.random.randint(0, num_feature)
        data[row][col] = np.nan
    return data
