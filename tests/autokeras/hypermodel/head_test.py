import tensorflow as tf

from autokeras.hypermodel import head as head_module
from tests import common


def test_y_is_pd_series():
    (x, y), (val_x, val_y) = common.dataframe_series()
    head = head_module.ClassificationHead()
    head.fit(y)
    assert isinstance(head.transform(y), tf.data.Dataset)
