from keras.layers import MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential

from autokeras.utils import *
from autokeras.search import *
import numpy as np


def test_model_trainer():
    model = RandomConvClassifierGenerator(3, (28, 28, 1)).generate()
    ModelTrainer(model, np.random.rand(2, 28, 28, 1), np.random.rand(2, 3), np.random.rand(1, 28, 28, 1),
                 np.random.rand(1, 3), False).train_model()

