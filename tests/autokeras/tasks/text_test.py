from unittest import mock

from autokeras.tasks import text
from tests import utils


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_text_classifier(auto_model, tmp_path):
    text.TextClassifier(directory=tmp_path, max_trials=2, seed=utils.SEED)
    assert auto_model.called


@mock.patch('autokeras.auto_model.AutoModel.__init__')
def test_text_regressor(auto_model, tmp_path):
    text.TextRegressor(directory=tmp_path, max_trials=2, seed=utils.SEED)
    assert auto_model.called
