from unittest import mock

import pytest

from autokeras.tasks import image
from tests import utils


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_image')


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_image_classifier(auto_model):
    image.ImageClassifier(directory=tmp_dir, max_trials=2, seed=utils.SEED)
    assert auto_model.called


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_image_regressor(auto_model):
    image.ImageRegressor(directory=tmp_dir, max_trials=2, seed=utils.SEED)
    assert auto_model.called
