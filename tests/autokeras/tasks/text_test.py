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

import autokeras as ak
from tests import utils


@mock.patch("autokeras.AutoModel.fit")
def test_txt_clf_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.TextClassifier(directory=tmp_path, seed=utils.SEED)

    auto_model.fit(x=np.array(["a b c", "b b c"]), y=np.array([1, 2]))

    assert fit.is_called


@mock.patch("autokeras.AutoModel.fit")
def test_txt_reg_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.TextRegressor(directory=tmp_path, seed=utils.SEED)

    auto_model.fit(x=np.array(["a b c", "b b c"]), y=np.array([1.0, 2.0]))

    assert fit.is_called
