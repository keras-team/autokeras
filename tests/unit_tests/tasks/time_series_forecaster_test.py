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

import autokeras as ak
from tests import utils


@mock.patch("autokeras.AutoModel.fit")
@mock.patch("autokeras.AutoModel.evaluate")
def test_tsf_evaluate_call_automodel_evaluate(evaluate, fit, tmp_path):
    auto_model = ak.TimeseriesForecaster(
        lookback=10, directory=tmp_path, seed=utils.SEED
    )

    auto_model.fit(x=utils.TRAIN_CSV_PATH, y="survived")
    auto_model.evaluate(x=utils.TRAIN_CSV_PATH, y="survived")

    assert evaluate.is_called


@mock.patch("autokeras.AutoModel.fit")
@mock.patch("autokeras.AutoModel.predict")
def test_tsf_predict_call_automodel_predict(predict, fit, tmp_path):
    auto_model = ak.TimeseriesForecaster(
        lookback=10, directory=tmp_path, seed=utils.SEED
    )

    auto_model.fit(x=utils.TRAIN_CSV_PATH, y="survived")
    auto_model.predict(x=utils.TRAIN_CSV_PATH, y="survived")

    assert predict.is_called


@mock.patch("autokeras.AutoModel.fit")
def test_tsf_fit_call_automodel_fit(fit, tmp_path):
    auto_model = ak.TimeseriesForecaster(
        lookback=10, directory=tmp_path, seed=utils.SEED
    )

    auto_model.fit(
        x=utils.TRAIN_CSV_PATH,
        y="survived",
        validation_data=(utils.TRAIN_CSV_PATH, "survived"),
    )

    assert fit.is_called
