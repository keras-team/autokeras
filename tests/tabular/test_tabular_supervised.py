import pytest
import numpy as np

from autokeras.tabular.tabular_supervised import *

from tests.common import clean_dir, TEST_TEMP_DIR


def test_fit_evalute_predict_binary_classification():
    train_x = None
    train_y = None
    clean_dir(TEST_TEMP_DIR)
    clf = TabularClassifier(path=TEST_TEMP_DIR)
    nsample = 59
    feat_exist_ind = [0, 5]
    for ntime in feat_exist_ind:
        for nnum in feat_exist_ind:
            for ncat in feat_exist_ind:
                    datainfo = np.array(['TIME'] * ntime + ['NUM'] * nnum + ['CAT'] * ncat)
                    num_feat = ntime + nnum + ncat
                    x_num = np.random.random([nsample, nnum])
                    x_time = np.random.random([nsample, ntime])
                    x_cat = np.random.randint(0, 40, [nsample, ncat])

                    train_x = np.concatenate([x_num, x_time, x_cat], axis=1)
                    train_y = np.random.randint(0, 2, nsample)
                    print(datainfo)
                    if num_feat == 0:
                        with pytest.raises(ValueError):
                            clf.fit(train_x, train_y, data_info=datainfo)
                    else:
                        clf.fit(train_x, train_y, data_info=datainfo)
                        results = clf.predict(train_x)
                        assert all(map(lambda result: result in train_y, results))
    score = clf.evaluate(train_x, train_y)
    assert score >= 0.0
    clean_dir(TEST_TEMP_DIR)


def test_fit_evalute_predict_multiclass_classification():
    clean_dir(TEST_TEMP_DIR)
    clf = TabularClassifier(path=TEST_TEMP_DIR)
    clf.verbose = True
    nsample = 1000
    [ntime, nnum, ncat] = [11, 15, 13]
    datainfo = np.array(['TIME'] * ntime + ['NUM'] * nnum + ['CAT'] * ncat)
    x_num = np.random.random([nsample, nnum])
    x_time = np.random.random([nsample, ntime])
    x_cat = np.random.randint(0, 200, [nsample, ncat])
    train_x = np.concatenate([x_num, x_time, x_cat], axis=1)
    train_y = np.random.randint(0, 3, nsample)
    clf.fit(train_x, train_y, data_info=datainfo)
    results = clf.predict(train_x)
    assert all(map(lambda result: result in train_y, results))
    score = clf.evaluate(train_x, train_y)
    assert score >= 0.0
    clean_dir(TEST_TEMP_DIR)


def test_fit_predict_evalute_regression():
    clean_dir(TEST_TEMP_DIR)
    clf = TabularRegressor(path=TEST_TEMP_DIR)
    nsample = 1000
    [ntime, nnum, ncat] = [3, 15, 3]
    datainfo = np.array(['TIME'] * ntime + ['NUM'] * nnum + ['CAT'] * ncat)
    x_num = np.random.random([nsample, nnum])
    x_time = np.random.random([nsample, ntime])
    x_cat = np.random.randint(0, 200, [nsample, ncat])
    train_x = np.concatenate([x_num, x_time, x_cat], axis=1)
    train_y = train_x[:, 5]
    clf.fit(train_x, train_y, data_info=datainfo)
    results = clf.predict(train_x)
    assert len(results) == len(train_x)
    score = clf.evaluate(train_x, train_y)
    assert score >= 0.0
    clean_dir(TEST_TEMP_DIR)
