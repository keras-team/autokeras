from unittest import mock

import autokeras as ak
from tests import utils


@mock.patch('autokeras.AutoModel.fit')
def test_img_clf_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.ImageClassifier(directory=tmp_path, seed=utils.SEED)

    auto_model.fit(
        x=utils.generate_data(num_instances=100, shape=(32, 32, 3)),
        y=utils.generate_one_hot_labels(num_instances=100, num_classes=10))

    assert fit.is_called


@mock.patch('autokeras.AutoModel.fit')
def test_img_reg_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.ImageRegressor(directory=tmp_path, seed=utils.SEED)

    auto_model.fit(
        x=utils.generate_data(num_instances=100, shape=(32, 32, 3)),
        y=utils.generate_data(num_instances=100, shape=(1,)))

    assert fit.is_called


@mock.patch('autokeras.AutoModel.fit')
def test_img_seg_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.tasks.image.ImageSegmenter(
        directory=tmp_path, seed=utils.SEED)

    auto_model.fit(
        x=utils.generate_data(num_instances=100, shape=(32, 32, 3)),
        y=utils.generate_data(num_instances=100, shape=(32, 32)))

    assert fit.is_called
