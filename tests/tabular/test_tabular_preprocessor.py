import pytest

from autokeras.tabular.tabular_preprocessor import *

from tests.common import clean_dir, TEST_TEMP_DIR


def test_extract_data_info():
    clean_dir(TEST_TEMP_DIR)
    tp = TabularPreprocessor()
    nsample = 10000
    half_sample = 5000
    [tp.n_num, tp.n_cat] = [5, 1]
    x_num = np.random.random([nsample, tp.n_num])
    x_cat = np.array(['a'] * half_sample + ['b'] * half_sample).reshape([nsample, 1])
    raw_x = np.concatenate([x_num, x_cat], axis=1)
    data_info = tp.extract_data_info(raw_x)
    ground_truth = np.array(['NUM'] * tp.n_num + ['CAT'] * tp.n_cat)
    assert (data_info == ground_truth).all()
    clean_dir(TEST_TEMP_DIR)