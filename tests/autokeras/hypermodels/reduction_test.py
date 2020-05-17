import tensorflow as tf

from autokeras.hypermodels import reduction
from tests.autokeras.hypermodels import utils


def test_merge():
    utils.block_basic_exam(
        reduction.Merge(),
        [
            tf.keras.Input(shape=(32,), dtype=tf.float32),
            tf.keras.Input(shape=(4, 8), dtype=tf.float32),
        ],
        ['merge_type'],
    )


def test_temporal_reduction():
    utils.block_basic_exam(
        reduction.TemporalReduction(),
        tf.keras.Input(shape=(32, 10), dtype=tf.float32),
        ['reduction_type'],
    )


def test_spatial_reduction():
    utils.block_basic_exam(
        reduction.SpatialReduction(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
        ['reduction_type'],
    )
