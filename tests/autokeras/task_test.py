from unittest import mock

import numpy as np
import pandas
import pytest

import autokeras as ak
from autokeras import task
from tests import common


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_image')


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_image_classifier(auto_model):
    task.ImageClassifier(directory=tmp_dir, max_trials=2, seed=common.SEED)
    assert auto_model.called


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_image_regressor(auto_model):
    task.ImageRegressor(directory=tmp_dir, max_trials=2, seed=common.SEED)
    assert auto_model.called


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_text_classifier(auto_model):
    task.TextClassifier(directory=tmp_dir, max_trials=2, seed=common.SEED)
    assert auto_model.called


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_text_regressor(auto_model):
    task.TextRegressor(directory=tmp_dir, max_trials=2, seed=common.SEED)
    assert auto_model.called


def test_structured_data_unknown_str_in_col_type(tmp_dir):
    with pytest.raises(ValueError) as info:
        task.StructuredDataClassifier(
            column_types=common.FALSE_COLUMN_TYPES_FROM_CSV,
            directory=tmp_dir,
            max_trials=1,
            seed=common.SEED)
    assert 'Column_types should be either "categorical"' in str(info.value)


def test_structured_data_col_name_type_mismatch(tmp_dir):
    with pytest.raises(ValueError) as info:
        task.StructuredDataClassifier(
            column_names=common.COLUMN_NAMES_FROM_NUMPY,
            column_types=common.COLUMN_TYPES_FROM_CSV,
            directory=tmp_dir,
            max_trials=1,
            seed=common.SEED)
    assert 'Column_names and column_types are mismatched.' in str(info.value)


@mock.patch('autokeras.auto_model.AutoModel.fit')
@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_structured_classifier(init, fit):
    num_data = 500
    train_x = common.generate_structured_data(num_data)
    train_y = common.generate_one_hot_labels(num_instances=num_data, num_classes=3)

    clf = ak.StructuredDataClassifier(
        column_names=common.COLUMN_NAMES_FROM_NUMPY,
        directory=tmp_dir,
        max_trials=1,
        seed=common.SEED)
    clf.fit(train_x, train_y, epochs=2, validation_data=(train_x, train_y))

    assert init.called
    assert fit.called


@mock.patch('autokeras.auto_model.AutoModel.fit')
@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_structured_regressor(init, fit):
    num_data = 500
    train_x = common.generate_structured_data(num_data)
    train_y = common.generate_data(num_instances=100, shape=(1,))

    clf = ak.StructuredDataRegressor(
        column_names=common.COLUMN_NAMES_FROM_NUMPY,
        directory=tmp_dir,
        max_trials=1,
        seed=common.SEED)
    clf.fit(train_x, train_y, epochs=2, validation_data=(train_x, train_y))

    assert init.called
    assert fit.called


@mock.patch('autokeras.auto_model.AutoModel.fit')
@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_structured_data_classifier_from_csv(init, fit):
    clf = ak.StructuredDataClassifier(directory=tmp_dir,
                                      max_trials=1,
                                      seed=common.SEED)

    clf.fit(x=common.TRAIN_FILE_PATH, y='survived', epochs=2,
            validation_data=common.TEST_FILE_PATH)

    assert init.called
    _, kwargs = fit.call_args_list[0]
    assert isinstance(kwargs['x'], pandas.DataFrame)
    assert isinstance(kwargs['y'], np.ndarray)
