import pytest

from autokeras.tabular.tabular_supervised import *

from tests.common import clean_dir, TEST_TEMP_DIR


def test_fit__evalute_predict_classification():
    # binary classification
    clean_dir(TEST_TEMP_DIR)
    clf = TabularClassifier(path=TEST_TEMP_DIR)
    nsample = 59
    feat_exist_ind = [0, 5]
    for ntime in feat_exist_ind:
        for nnum in feat_exist_ind:
            for ncat in feat_exist_ind:
                for nmvc in feat_exist_ind:
                    datainfo = {'loaded_feat_types': [ntime, nnum, ncat, nmvc]}
                    num_feat = ntime + nnum + ncat + nmvc
                    x_num = np.random.random([nsample, nnum])
                    x_time = np.random.random([nsample, ntime])
                    x_cat = np.random.randint(0, 40, [nsample, ncat])
                    x_mvc = np.random.randint(0, 10, [nsample, nmvc])

                    train_x = np.concatenate([x_num, x_time, x_cat, x_mvc], axis=1)
                    train_y = np.random.randint(0, 2, nsample)
                    print(datainfo)
                    if num_feat == 0:
                        with pytest.raises(ValueError):
                            clf.fit(train_x, train_y, datainfo=datainfo)
                    else:
                        clf.fit(train_x, train_y, datainfo=datainfo)
                        results = clf.predict(train_x)
                        assert all(map(lambda result: result in train_y, results))
    score = clf.evaluate(train_x, train_y)
    assert score >= 0.0

    # multiclass:
    nsample = 10000
    [ntime, nnum, ncat, nmvc] = [3, 15, 3, 3]
    datainfo = {'loaded_feat_types': [ntime, nnum, ncat, nmvc]}
    x_num = np.random.random([nsample, nnum])
    x_time = np.random.random([nsample, ntime])
    x_cat = np.random.randint(0, 200, [nsample, ncat])
    x_mvc = np.random.randint(0, 10, [nsample, nmvc])
    train_x = np.concatenate([x_num, x_time, x_cat, x_mvc], axis=1)
    train_y = np.random.randint(0, 3, nsample)
    clf.fit(train_x, train_y, datainfo=datainfo)
    results = clf.predict(train_x)
    assert all(map(lambda result: result in train_y, results))
    score = clf.evaluate(train_x, train_y)
    assert score >= 0.0
    clean_dir(TEST_TEMP_DIR)


def test_fit_predict_evalute_regression():
    clean_dir(TEST_TEMP_DIR)
    clf = TabularRegressor(path=TEST_TEMP_DIR)
    nsample = 10000
    [ntime, nnum, ncat, nmvc] = [3, 15, 3, 3]
    datainfo = {'loaded_feat_types': [ntime, nnum, ncat, nmvc]}
    x_num = np.random.random([nsample, nnum])
    x_time = np.random.random([nsample, ntime])
    x_cat = np.random.randint(0, 200, [nsample, ncat])
    x_mvc = np.random.randint(0, 10, [nsample, nmvc])
    train_x = np.concatenate([x_num, x_time, x_cat, x_mvc], axis=1)
    train_y = train_x[:, 5]
    clf.fit(train_x, train_y, datainfo=datainfo)
    results = clf.predict(train_x)
    assert len(results) == len(train_x)
    score = clf.evaluate(train_x, train_y)
    assert score >= 0.0
    # test different model loading in predict
    clf.clf = None
    results = clf.predict(train_x)
    assert len(results) == len(train_x)
    clf.clf = None
    clf.save_filename = None
    results = clf.predict(train_x)
    assert len(results) == len(train_x)
    clean_dir(TEST_TEMP_DIR)
