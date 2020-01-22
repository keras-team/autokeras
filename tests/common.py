import kerastuner
import numpy as np
import pandas as pd
import tensorflow as tf

import autokeras as ak

SEED = 5
COLUMN_NAMES_FROM_NUMPY = [
    'bool_',
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
    'sex',
    'age',
    'n_siblings_spouses',
    'parch',
    'fare',
    'class',
    'deck',
    'embark_town',
    'alone']
LESS_COLUMN_NAMES_FROM_CSV = [
    'age',
    'n_siblings_spouses',
    'parch',
    'fare',
    'class',
    'deck',
    'embark_town',
    'alone']
COLUMN_TYPES_FROM_CSV = {
    'sex': 'categorical',
    'age': 'numerical',
    'n_siblings_spouses': 'categorical',
    'parch': 'categorical',
    'fare': 'numerical',
    'class': 'categorical',
    'deck': 'categorical',
    'embark_town': 'categorical',
    'alone': 'categorical'}
FALSE_COLUMN_TYPES_FROM_CSV = {
    'sex': 'cat',
    'age': 'num',
    'n_siblings_spouses': 'cat',
    'parch': 'categorical',
    'fare': 'numerical',
    'class': 'categorical',
    'deck': 'categorical',
    'embark_town': 'categorical',
    'alone': 'categorical'}
PARTIAL_COLUMN_TYPES_FROM_CSV = {
    'fare': 'categorical',
    'class': 'categorical',
    'deck': 'categorical',
    'embark_town': 'categorical',
    'alone': 'categorical'}
TRAIN_FILE_PATH = r'tests/fixtures/titanic/train.csv'
TEST_FILE_PATH = r'tests/fixtures/titanic/eval.csv'


def generate_structured_data(num_instances=500, dtype='np'):
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
    col_bool = np.random.choice(boolean, num_instances).reshape(num_instances, 1)
    col_num_to_cat = np.random.randint(
        20, 41, size=num_instances).reshape(num_instances, 1)
    col_float = 100*np.random.random(num_instances,).reshape(num_instances, 1)
    col_int = np.random.randint(
        2000, 4000, size=num_instances).reshape(num_instances, 1)
    col_morethan_32 = np.random.choice(
        career_size, num_instances).reshape(num_instances, 1)
    col1_morethan_100 = np.random.choice(career_states,
                                         num_instances).reshape(num_instances, 1)
    col2_morethan_100 = np.random.choice(career_years,
                                         num_instances).reshape(num_instances, 1)
    col3_morethan_100 = np.random.choice(career_color,
                                         num_instances).reshape(num_instances, 1)
    data = np.concatenate((col_bool, col_num_to_cat, col_float, col_int,
                           col_morethan_32, col1_morethan_100, col2_morethan_100,
                           col3_morethan_100), axis=1)
    # generate np.nan data
    for i in range(num_nan):
        row = np.random.randint(0, num_instances)
        col = np.random.randint(0, num_feature)
        data[row][col] = np.nan
    if dtype == 'np':
        return data
    if dtype == 'dataset':
        return tf.data.Dataset.from_tensor_slices(data)


def dataframe_numpy():
    x = pd.read_csv(TRAIN_FILE_PATH)
    y = x.pop('survived').to_numpy()
    val_x = pd.read_csv(TEST_FILE_PATH)
    val_y = val_x.pop('survived').to_numpy()
    return (x, y), (val_x, val_y)


def dataframe_dataframe():
    x = pd.read_csv(TRAIN_FILE_PATH)
    y = pd.DataFrame(x.pop('survived'))
    val_x = pd.read_csv(TEST_FILE_PATH)
    val_y = pd.DataFrame(val_x.pop('survived'))
    return (x, y), (val_x, val_y)


def dataframe_series():
    x = pd.read_csv(TRAIN_FILE_PATH)
    y = x.pop('survived')
    val_x = pd.read_csv(TEST_FILE_PATH)
    val_y = val_x.pop('survived')
    return (x, y), (val_x, val_y)


def csv_test(target):
    x_test = pd.read_csv(TEST_FILE_PATH)
    if target == 'regression':
        x_test = x_test.drop('fare', axis=1)
    else:
        x_test = x_test.drop('survived', axis=1)
    return x_test


def generate_data(num_instances=100, shape=(32, 32, 3), dtype='np'):
    np.random.seed(SEED)
    data = np.random.rand(*((num_instances,) + shape))
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    if dtype == 'np':
        return data
    if dtype == 'dataset':
        return tf.data.Dataset.from_tensor_slices(data)


def generate_one_hot_labels(num_instances=100, num_classes=10, dtype='np'):
    np.random.seed(SEED)
    labels = np.random.randint(num_classes, size=num_instances)
    data = tf.keras.utils.to_categorical(labels)
    if dtype == 'np':
        return data
    if dtype == 'dataset':
        return tf.data.Dataset.from_tensor_slices(data)


def fit_predict_with_graph(inputs, outputs, x, y):
    model = ak.hypermodel.graph.HyperBuiltGraphHyperModel(
        inputs, outputs).build(kerastuner.HyperParameters())
    model.fit(x, y,
              epochs=1,
              batch_size=100,
              verbose=False,
              validation_split=0.2)
    return model.predict(x)


def do_nothing(*args, **kwargs):
    pass


def imdb_raw(num_instances=100):
    index_offset = 3  # word index offset

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=1000,
        index_from=index_offset)
    x_train = x_train[:num_instances]
    y_train = y_train[:num_instances].reshape(-1, 1)
    x_test = x_test[:num_instances]
    y_test = y_test[:num_instances].reshape(-1, 1)

    word_to_id = tf.keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}
    x_train = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_train))
    x_test = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_test))
    x_train = np.array(x_train, dtype=np.str)
    x_test = np.array(x_test, dtype=np.str)
    return (x_train, y_train), (x_test, y_test)


def name_in_hps(hp_name, hp):
    return any([hp_name in name for name in hp.values])


def build_graph():
    tf.keras.backend.clear_session()
    image_input = ak.ImageInput(shape=(32, 32, 3))
    merged_outputs = ak.ImageBlock()(image_input)
    head = ak.ClassificationHead(num_classes=10)
    head.output_shape = (10,)
    classification_outputs = head(merged_outputs)
    return ak.hypermodel.graph.Graph(
        inputs=image_input,
        outputs=classification_outputs)
