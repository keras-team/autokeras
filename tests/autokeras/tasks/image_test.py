from unittest import mock

from autokeras.tasks import image
from tests import utils


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_image_classifier(auto_model, tmp_path):
    image.ImageClassifier(directory=tmp_path, max_trials=2, seed=utils.SEED)
    assert auto_model.called


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_image_regressor(auto_model, tmp_path):
    image.ImageRegressor(directory=tmp_path, max_trials=2, seed=utils.SEED)
    assert auto_model.called


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_image_segmenter(auto_model, tmp_path):
    image.ImageSegmenter(directory=tmp_path, max_trials=2, seed=utils.SEED)
    assert auto_model.called
