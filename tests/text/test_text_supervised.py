from unittest.mock import patch
from tests.common import MockProcess


def mock_train(**kwargs):
    str(kwargs)
    return 1, 0


def mock_text_preprocess(x_train):
    return x_train


def test_fit_predict():
    pass


def test_evaluate():
    pass
