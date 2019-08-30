import functools
import kerastuner
import numpy as np
import pytest
import tensorflow as tf

import autokeras as ak
from autokeras.hypermodel import preprocessor
from autokeras.hypermodel import block
from autokeras.hypermodel import head


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_lgbm')


def test_normalize():
    normalize = preprocessor.Normalization()
    x_train = np.random.rand(100, 32, 32, 3)
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    normalize.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        normalize.update(x)
    normalize.finalize()
    normalize.set_config(normalize.get_config())

    weights = normalize.get_weights()
    normalize.clear_weights()
    normalize.set_weights(weights)

    for a in dataset:
        normalize.transform(a)

    def map_func(x):
        return tf.py_function(normalize.transform,
                              inp=[x],
                              Tout=(tf.float64,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
    assert isinstance(new_dataset, tf.data.Dataset)


def test_sequence():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    tokenize = preprocessor.TextToIntSequence()
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    tokenize.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        tokenize.update(x)
    tokenize.finalize()
    tokenize.set_config(tokenize.get_config())

    weights = tokenize.get_weights()
    tokenize.clear_weights()
    tokenize.set_weights(weights)

    for a in dataset:
        tokenize.transform(a)

    def map_func(x):
        return tf.py_function(tokenize.transform,
                              inp=[x],
                              Tout=(tf.int64,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
    assert isinstance(new_dataset, tf.data.Dataset)


def test_ngram():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    tokenize = preprocessor.TextToNgramVector()
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    tokenize.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        tokenize.update(x)
    tokenize.finalize()
    tokenize.set_config(tokenize.get_config())

    weights = tokenize.get_weights()
    tokenize.clear_weights()
    tokenize.set_weights(weights)

    for a in dataset:
        tokenize.transform(a)

    def map_func(x):
        return tf.py_function(tokenize.transform,
                              inp=[x],
                              Tout=(tf.float64,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
    assert isinstance(new_dataset, tf.data.Dataset)


def test_augment():
    raw_images = tf.random.normal([1000, 32, 32, 3], mean=-1, stddev=4)
    augment = preprocessor.ImageAugmentation(seed=5)
    dataset = tf.data.Dataset.from_tensor_slices(raw_images)
    hp = kerastuner.HyperParameters()
    augment.set_hp(hp)
    augment.set_config(augment.get_config())
    for a in dataset:
        augment.transform(a, True)

    def map_func(x):
        return tf.py_function(functools.partial(augment.transform, fit=True),
                              inp=[x],
                              Tout=(tf.float32,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
    assert isinstance(new_dataset, tf.data.Dataset)


def test_feature_engineering():
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

    np.random.seed(0)
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
    feature = preprocessor.FeatureEngineering([
        'categorical', 'categorical', 'numerical', 'numerical', 'categorical',
        'categorical', 'categorical', 'categorical'])
    feature.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        feature.update(x)
    feature.finalize()
    feature.set_config(feature.get_config())
    for a in dataset:
        feature.transform(a)

    def map_func(x):
        return tf.py_function(feature.transform,
                              inp=[x],
                              Tout=(tf.float64,))
    new_dataset = dataset.map(map_func)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_feature_engineering_fix_keyerror():
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

    np.random.seed(0)
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
    feature = preprocessor.FeatureEngineering([
        'categorical', 'categorical', 'numerical', 'numerical', 'categorical',
        'categorical', 'categorical', 'categorical'])
    feature.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        feature.update(x)
    feature.finalize()
    feature.set_config(feature.get_config())
    for a in dataset:
        feature.transform(a)

    def map_func(x):
        return tf.py_function(feature.transform,
                              inp=[x],
                              Tout=(tf.float64,))
    new_dataset = dataset.map(map_func)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_lgbm_classifier(tmp_dir):
    x_train = np.random.rand(11, 32)
    y_train = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])

    input_node = ak.Input()
    output_node = input_node
    output_node = preprocessor.LightGBMClassifier()(output_node)
    output_node = block.IdentityBlock()(output_node)
    output_node = head.EmptyHead(loss='categorical_crossentropy',
                                 metrics=['accuracy'])(output_node)

    auto_model = ak.GraphAutoModel(input_node,
                                   output_node,
                                   directory=tmp_dir,
                                   max_trials=1)
    auto_model.fit(x_train, y_train, epochs=1,
                   validation_data=(x_train, y_train))
    result = auto_model.predict(x_train)
    auto_model.tuner.get_best_models()[0].summary()
    assert result.shape == (11, 10)


def test_lgbm_regressor(tmp_dir):
    x_train = np.random.rand(11, 32)
    y_train = np.array([1.1, 2.1, 4.2, 0.3, 2.4, 8.5, 7.3, 8.4, 9.4, 4.3])
    input_node = ak.Input()
    output_node = input_node
    output_node = preprocessor.LightGBMRegressor()(output_node)
    output_node = block.IdentityBlock()(output_node)
    output_node = head.EmptyHead(loss='mean_squared_error',
                                 metrics=['mean_squared_error'])(output_node)

    auto_model = ak.GraphAutoModel(input_node,
                                   output_node,
                                   directory=tmp_dir,
                                   max_trials=1)
    auto_model.fit(x_train, y_train, epochs=1,
                   validation_data=(x_train, y_train))
    result = auto_model.predict(x_train)
    auto_model.tuner.get_best_models()[0].summary()
    assert result.shape == (11, 1)
