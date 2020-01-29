from unittest import mock

import pytest

from autokeras.tasks import text
from tests import utils


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_text')


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_text_classifier(auto_model, tmp_dir):
    text.TextClassifier(directory=tmp_dir, max_trials=2, seed=utils.SEED)
    assert auto_model.called


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_text_regressor(auto_model, tmp_dir):
    text.TextRegressor(directory=tmp_dir, max_trials=2, seed=utils.SEED)
    assert auto_model.called
