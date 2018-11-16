from unittest.mock import patch

import numpy as np

from autokeras.constant import Constant
from autokeras.image.gan import DCGAN
from tests.common import clean_dir, TEST_TEMP_DIR


def mock_train(**kwargs):
    str(kwargs)
    return 1, 0


@patch('autokeras.image.gan.GANModelTrainer.train_model', side_effect=mock_train)
def test_fit_generate(_):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    Constant.DATA_AUGMENTATION = False
    image_path, size = TEST_TEMP_DIR, 32
    clean_dir(image_path)
    dcgan = DCGAN(gen_training_result=(image_path, size))
    train_x = np.random.rand(100, 32, 32, 3)
    dcgan.fit(train_x)
    clean_dir(image_path)
    noise = np.random.randn(32, 100, 1, 1).astype('float32')
    dcgan.generate(noise)
