import copy
import pytest
import tensorflow as tf
import numpy as np
import pandas as pd

from autokeras.analysers import input_analysers
from tests import utils


def test_structured_data_input_less_col_name_error():
    with pytest.raises(ValueError) as info:
        analyser = input_analysers.StructuredDataAnalyser(
            column_names=list(range(8))
        )
        dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(20, 10)).batch(32)
        for x in dataset:
            analyser.update(x)

        analyser.finalize()

    assert "Expect column_names to have length" in str(info.value)


def test_structured_data_infer_col_types():
    analyser = input_analysers.StructuredDataAnalyser(
        column_names=utils.COLUMN_NAMES,
        column_types=None,
    )
    x = pd.read_csv(utils.TRAIN_CSV_PATH)
    x.pop("survived")
    dataset = tf.data.Dataset.from_tensor_slices(x.values.astype(np.unicode)).batch(32)

    for data in dataset:
        analyser.update(data)
    analyser.finalize()

    assert analyser.column_types == utils.COLUMN_TYPES

def test_dont_infer_specified_column_types():
    column_types = copy.copy(utils.COLUMN_TYPES)
    column_types.pop("sex")
    column_types["age"] = "categorical"

    analyser = input_analysers.StructuredDataAnalyser(
        column_names=utils.COLUMN_NAMES,
        column_types=column_types,
    )
    x = pd.read_csv(utils.TRAIN_CSV_PATH)
    x.pop("survived")
    dataset = tf.data.Dataset.from_tensor_slices(x.values.astype(np.unicode)).batch(32)

    for data in dataset:
        analyser.update(data)
    analyser.finalize()

    assert analyser.column_types["age"] == "categorical"
