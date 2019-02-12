from unittest.mock import patch
from tests.common import MockProcess


def mock_train(**kwargs):
    str(kwargs)
    return 1, 0


def mock_text_preprocess(x_train):
    return x_train


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_predict(_, _1, _2):
    pass


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
@patch('autokeras.text.text_supervised.text_preprocess', side_effect=mock_text_preprocess)
def test_evaluate(_, _1, _2):
    pass
