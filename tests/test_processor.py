import tensorflow as tf
import numpy as np
import kerastuner

from autokeras.hypermodel import processor


def test_normalize():
    normalize = processor.Normalize()
    x_train = np.random.rand(100, 32, 32, 3)
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    new_dataset = normalize.fit_transform(kerastuner.HyperParameters(), dataset)
    assert isinstance(new_dataset, tf.data.Dataset)
