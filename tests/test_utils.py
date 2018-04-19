from unittest.mock import patch
from autokeras.generator import RandomConvClassifierGenerator
from autokeras.utils import *
import numpy as np


def test_model_trainer():
    model = RandomConvClassifierGenerator(3, (28, 28, 1)).generate()
    ModelTrainer(model, np.random.rand(2, 28, 28, 1), np.random.rand(2, 3), np.random.rand(1, 28, 28, 1),
                 np.random.rand(1, 3), False).train_model()


@patch('autokeras.utils.backend')
def test_model_trainer_not_augmented(_):
    constant.DATA_AUGMENTATION = False
    constant.LIMIT_MEMORY = True
    model = RandomConvClassifierGenerator(3, (28, 28, 1)).generate()
    ModelTrainer(model, np.random.rand(2, 28, 28, 1), np.random.rand(2, 3), np.random.rand(1, 28, 28, 1),
                 np.random.rand(1, 3), False).train_model()
