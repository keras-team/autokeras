from unittest import mock

import numpy as np
import pandas
import pytest

from autokeras.tasks import structured_data
from tests import utils


def test_structured_data_unknown_str_in_col_type(tmp_path):
    with pytest.raises(ValueError) as info:
        structured_data.StructuredDataClassifier(
            column_types=utils.FALSE_COLUMN_TYPES_FROM_CSV,
            directory=tmp_path,
            max_trials=1,
            seed=utils.SEED)
    assert 'Column_types should be either "categorical"' in str(info.value)


def test_structured_data_col_name_type_mismatch(tmp_path):
    with pytest.raises(ValueError) as info:
        structured_data.StructuredDataClassifier(
            column_names=utils.COLUMN_NAMES_FROM_NUMPY,
            column_types=utils.COLUMN_TYPES_FROM_CSV,
            directory=tmp_path,
            max_trials=1,
            seed=utils.SEED)
    assert 'Column_names and column_types are mismatched.' in str(info.value)


@mock.patch('autokeras.auto_model.AutoModel.fit')
@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_structured_classifier(init, fit, tmp_path):
    num_data = 500
    train_x = utils.generate_structured_data(num_data)
    train_y = utils.generate_one_hot_labels(num_instances=num_data, num_classes=3)

    clf = structured_data.StructuredDataClassifier(
        column_names=utils.COLUMN_NAMES_FROM_NUMPY,
        directory=tmp_path,
        max_trials=1,
        seed=utils.SEED)
    clf.fit(train_x, train_y, epochs=2, validation_data=(train_x, train_y))

    assert init.called
    assert fit.called


@mock.patch('autokeras.auto_model.AutoModel.fit')
@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_structured_regressor(init, fit, tmp_path):
    num_data = 500
    train_x = utils.generate_structured_data(num_data)
    train_y = utils.generate_data(num_instances=100, shape=(1,))

    clf = structured_data.StructuredDataRegressor(
        column_names=utils.COLUMN_NAMES_FROM_NUMPY,
        directory=tmp_path,
        max_trials=1,
        seed=utils.SEED)
    clf.fit(train_x, train_y, epochs=2, validation_data=(train_x, train_y))

    assert init.called
    assert fit.called


@mock.patch('autokeras.auto_model.AutoModel.fit')
@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_structured_data_classifier_from_csv(init, fit, tmp_path):
    clf = structured_data.StructuredDataClassifier(
        directory=tmp_path,
        max_trials=1,
        seed=utils.SEED)

    clf.fit(x=utils.TRAIN_FILE_PATH, y='survived', epochs=2,
            validation_data=(utils.TEST_FILE_PATH, 'survived'))

    assert init.called
    _, kwargs = fit.call_args_list[0]
    assert isinstance(kwargs['x'], pandas.DataFrame)
    assert isinstance(kwargs['y'], np.ndarray)
