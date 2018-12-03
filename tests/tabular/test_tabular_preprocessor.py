import pytest

from autokeras.tabular.tabular_preprocessor import *

from tests.common import clean_dir, TEST_TEMP_DIR

def test_extract_data():
    clean_dir(TEST_TEMP_DIR)
    tp = TabularPreprocessor()
    nsample = 10000
    [tp.ntime, tp.nnum, tp.ncat, tp.nmvc] = [3, 15, 3, 3]
    num_feat = tp.ntime + tp.nnum + tp.ncat + tp.nmvc
    [tp.ntime, tp.nnum, tp.ncat, tp.nmvc] = [3, 15, 3, 3]
    x_num = np.random.random([nsample, tp.nnum])
    x_time = np.random.random([nsample, tp.ntime])
    x_cat = np.random.randint(0, 200, [nsample, tp.ncat])
    x_mvc = np.random.randint(0, 10, [nsample, tp.nmvc])
    F = {'numerical': np.concatenate((x_time, x_num), axis=1),
         'CAT': x_cat,
         'MV': x_mvc}
    for col_index in range(tp.nnum + tp.ntime, num_feat):
        tp.cat_to_int_label[col_index] = {}

    x = tp.extract_data(F, tp.ncat, tp.nmvc)
    assert x.shape == (nsample, num_feat)
    clean_dir(TEST_TEMP_DIR)

def test_encode():
    clean_dir(TEST_TEMP_DIR)
    tp = TabularPreprocessor()
    nsample = 10000
    [tp.ntime, tp.nnum, tp.ncat, tp.nmvc] = [3, 15, 3, 3]
    num_feat = tp.ntime + tp.nnum + tp.ncat + tp.nmvc
    x_num = np.random.random([nsample, tp.nnum])
    x_time = np.random.random([nsample, tp.ntime])
    x_cat = np.random.randint(0, 200, [nsample, tp.ncat])
    x_mvc = np.random.randint(0, 10, [nsample, tp.nmvc])
    train_x = np.concatenate([x_num, x_time, x_cat, x_mvc], axis=1)
    x = tp.encode(train_x)
    assert x.shape == (nsample, num_feat)
    clean_dir(TEST_TEMP_DIR)
