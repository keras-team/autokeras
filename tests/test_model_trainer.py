from autokeras.gan import Generator, Discriminator
from autokeras.generator import CnnGenerator
from autokeras.loss_function import classification_loss, regression_loss, binary_classification_loss
from autokeras.metric import Accuracy, MSE
from autokeras.model_trainer import ModelTrainer, GANModelTrainer
from tests.common import get_classification_data_loaders, get_regression_data_loaders, \
    get_classification_train_data_loaders


def test_model_trainer_classification():
    model = CnnGenerator(3, (28, 28, 3)).generate().produce_model()
    train_data, test_data = get_classification_data_loaders()
    ModelTrainer(model, train_data, test_data, Accuracy, classification_loss, True).train_model(max_iter_num=3)


def test_model_trainer_regression():
    model = CnnGenerator(1, (28, 28, 3)).generate().produce_model()
    train_data, test_data = get_regression_data_loaders()
    ModelTrainer(model, train_data, test_data, MSE, regression_loss, False).train_model(max_iter_num=3)


def test_gan_model_trainer():
    g_model = Generator(3, 100, 64)
    d_model = Discriminator(3, 64)
    train_data = get_classification_train_data_loaders()
    GANModelTrainer(g_model, d_model, train_data, binary_classification_loss, True).train_model(max_iter_num=3)
