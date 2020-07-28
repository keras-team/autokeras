# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

import numpy as np
import pandas
import pytest

import autokeras as ak
from tests import utils


def test_raise_error_unknown_str_in_col_type(tmp_path):
    with pytest.raises(ValueError) as info:
        ak.StructuredDataClassifier(
            column_types={'age': 'num', 'parch': 'categorical'},
            directory=tmp_path,
            seed=utils.SEED)

    assert 'Column_types should be either "categorical"' in str(info.value)


@mock.patch('autokeras.AutoModel.fit')
def test_structured_clf_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.StructuredDataClassifier(directory=tmp_path, seed=utils.SEED)

    auto_model.fit(
        x=utils.generate_structured_data(num_instances=100),
        y=utils.generate_one_hot_labels(num_instances=100, num_classes=3))

    assert fit.is_called


@mock.patch('autokeras.AutoModel.fit')
def test_structured_reg_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.StructuredDataRegressor(directory=tmp_path, seed=utils.SEED)

    auto_model.fit(
        x=utils.generate_structured_data(num_instances=100),
        y=utils.generate_data(num_instances=100, shape=(1,)))

    assert fit.is_called


@mock.patch('autokeras.AutoModel.fit')
def test_structured_data_clf_convert_csv_to_df_and_np(fit, tmp_path):
    auto_model = ak.StructuredDataClassifier(directory=tmp_path, seed=utils.SEED)

    auto_model.fit(x=utils.TRAIN_FILE_PATH,
                   y='survived',
                   epochs=2,
                   validation_data=(utils.TEST_FILE_PATH, 'survived'))

    _, kwargs = fit.call_args_list[0]
    assert isinstance(kwargs['x'], pandas.DataFrame)
    assert isinstance(kwargs['y'], np.ndarray)
