import os
import numpy as np
from keras import Input, Model
from keras.losses import mean_squared_error
from keras.models import load_model
from tensorflow.python.layers.utils import constant_value

from autokeras.layers import *
from tests.common import get_add_skip_model, clean_dir


def test_weighted_add():
    a1 = Input(shape=(3, 3, 2))
    a2 = Input(shape=(3, 3, 2))
    layer = WeightedAdd()
    b = layer([a1, a2])
    model = Model(inputs=[a1, a2], outputs=b)
    data = np.ones((1, 3, 3, 2))
    model.compile(optimizer='Adam', loss=mean_squared_error)
    model.fit([data, data * 2], data * 2, epochs=10, verbose=False)
    results = model.predict_on_batch([data, data * 2])
    assert not np.array_equal(results, data)
    assert constant_value(layer.one) == 1


def test_save_weighted_add():
    model = get_add_skip_model()
    path = 'tests/resources/temp/m.h5'
    model.save(path)
    load_model(path, {'WeightedAdd': WeightedAdd})
    os.remove(path)
