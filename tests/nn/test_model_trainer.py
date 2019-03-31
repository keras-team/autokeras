from autokeras.backend import Backend
from autokeras.image.gan import Generator, Discriminator, GANModelTrainer
from autokeras.nn.generator import CnnGenerator
from autokeras.nn.metric import Accuracy, MSE
from autokeras.backend.torch.model_trainer import ModelTrainer
from tests.common import get_classification_data_loaders, get_regression_data_loaders, \
    get_classification_train_data_loaders, clean_dir, TEST_TEMP_DIR
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


def test_gan_model_trainer():
    g_model = Generator(3, 100, 64)
    d_model = Discriminator(3, 64)
    train_data = get_classification_train_data_loaders()
    GANModelTrainer(g_model, d_model, train_data, Backend.binary_classification_loss, True).train_model(max_iter_num=3)


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
