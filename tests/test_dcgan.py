from unittest.mock import patch

import numpy as np

from autokeras.constant import Constant
from autokeras.gan import DCGAN
from tests.common import clean_dir


def mock_train(**kwargs):
    str(kwargs)
    return 1, 0


@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_train(_):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    Constant.DATA_AUGMENTATION = False
    image_path, size = 'tests/resources/temp', 32
    clean_dir(image_path)
    dcgan = DCGAN(gen_training_result=(image_path, size))
    train_x = np.random.rand(100, 32, 32, 3)
    dcgan.train(train_x)
    clean_dir(image_path)
