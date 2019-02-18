import os
import random

from unittest.mock import patch

from autokeras.text.text_supervised import TextClassifier
from tests.common import mock_bert_fit, mock_bert_predict


@patch('autokeras.text.text_supervised.TextClassifier.fit', side_effect=mock_bert_fit)
@patch('autokeras.text.text_supervised.TextClassifier.predict', side_effect=mock_bert_predict)
def test_fit(_, _1):
    clf = TextClassifier(verbose=True)

    train_x = ["sentence 1", "sentence 2"]
    train_y = ["0", "1"]
    clf.fit(train_x, train_y)

    test_x = train_x
    y_preds = clf.predict(test_x)
    if len(y_preds) != len(test_x):
        raise AssertionError()
