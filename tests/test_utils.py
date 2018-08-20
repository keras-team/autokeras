from unittest.mock import patch

from autokeras.generator import CnnGenerator
from autokeras.loss_function import classification_loss, regression_loss
from autokeras.metric import Accuracy, MSE
from autokeras.utils import ModelTrainer, temp_folder_generator
from tests.common import get_classification_data_loaders, get_regression_data_loaders, clean_dir


def test_model_trainer_classification():
    model = CnnGenerator(3, (28, 28, 3)).generate().produce_model()
    train_data, test_data = get_classification_data_loaders()
    ModelTrainer(model, train_data, test_data, Accuracy, classification_loss, True).train_model(max_iter_num=3)


def test_model_trainer_regression():
    model = CnnGenerator(1, (28, 28, 3)).generate().produce_model()
    train_data, test_data = get_regression_data_loaders()
    ModelTrainer(model, train_data, test_data, MSE, regression_loss, False).train_model(max_iter_num=3)


@patch('tempfile.gettempdir', return_value="tests/resources/temp/")
def test_temp_folder_generator(_):
    path = 'tests/resources/temp'
    clean_dir(path)
    path = temp_folder_generator()
    assert path == "tests/resources/temp/autokeras"
    path = 'tests/resources/temp'
    clean_dir(path)
