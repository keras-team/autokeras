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
def test_img_clf_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.ImageClassifier(directory=tmp_path, seed=utils.SEED)

    auto_model.fit(
        x=utils.generate_data(num_instances=100, shape=(32, 32, 3)),
        y=utils.generate_one_hot_labels(num_instances=100, num_classes=10),
    )

    assert fit.is_called


@mock.patch("autokeras.AutoModel.fit")
def test_img_reg_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.ImageRegressor(directory=tmp_path, seed=utils.SEED)

    auto_model.fit(
        x=utils.generate_data(num_instances=100, shape=(32, 32, 3)),
        y=utils.generate_data(num_instances=100, shape=(1,)),
    )

    assert fit.is_called


@mock.patch("autokeras.AutoModel.fit")
def test_img_seg_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.tasks.image.ImageSegmenter(directory=tmp_path, seed=utils.SEED)

    auto_model.fit(
        x=utils.generate_data(num_instances=100, shape=(32, 32, 3)),
        y=utils.generate_data(num_instances=100, shape=(32, 32)),
    )

    assert fit.is_called
