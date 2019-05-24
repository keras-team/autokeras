from autokeras.legacy.backend import Backend
from autokeras.legacy.backend.torch.model_trainer import ModelTrainer
from autokeras.legacy.nn.generator import CnnGenerator
from autokeras.legacy.nn.metric import MSE, Accuracy
from tests.legacy.common import get_classification_data_loaders, get_regression_data_loaders, \
    clean_dir, TEST_TEMP_DIR
import pytest


def test_model_trainer_classification():
    model = CnnGenerator(3, (28, 28, 3)).generate().produce_model()
    train_data, test_data = get_classification_data_loaders()
    ModelTrainer(model,
                 train_data=train_data,
                 test_data=test_data,
                 metric=Accuracy,
                 loss_function=Backend.classification_loss,
                 verbose=True,
                 path=TEST_TEMP_DIR).train_model(max_iter_num=3)
    clean_dir(TEST_TEMP_DIR)


def test_model_trainer_regression():
    model = CnnGenerator(1, (28, 28, 3)).generate().produce_model()
    train_data, test_data = get_regression_data_loaders()
    ModelTrainer(model,
                 train_data=train_data,
                 test_data=test_data,
                 metric=MSE,
                 loss_function=Backend.regression_loss,
                 verbose=False,
                 path=TEST_TEMP_DIR).train_model(max_iter_num=3)
    clean_dir(TEST_TEMP_DIR)


def test_model_trainer_timout():
    model = CnnGenerator(3, (28, 28, 3)).generate().produce_model()
    timeout = 1
    train_data, test_data = get_classification_data_loaders()
    with pytest.raises(TimeoutError):
        ModelTrainer(model,
                     train_data=train_data,
                     test_data=test_data,
                     metric=Accuracy,
                     loss_function=Backend.classification_loss,
                     verbose=True,
                     path=TEST_TEMP_DIR).train_model(max_iter_num=300, timeout=timeout)
    clean_dir(TEST_TEMP_DIR)
