import os
import numpy as np
from keras import Input, Model
from keras.losses import mean_squared_error
from keras.models import load_model
from tensorflow.python.layers.utils import constant_value

from autokeras.layers import *
from tests.common import get_add_skip_model, clean_dir


def test_save_weighted_add():
    model = get_add_skip_model()
    path = 'tests/resources/temp/m.h5'
    model.save(path)
    load_model(path)
    os.remove(path)
