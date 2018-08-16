from unittest.mock import patch

from autokeras.generator import CnnGenerator
from autokeras.loss_function import classification_loss
from autokeras.metric import Accuracy
from autokeras.utils import ModelTrainer, temp_folder_generator
from tests.common import get_processed_data


def test_model_trainer():
    model = CnnGenerator(3, (28, 28, 3)).generate().produce_model()
    train_data, test_data = get_processed_data()
    ModelTrainer(model, train_data, test_data, Accuracy, classification_loss, False).train_model(max_iter_num=3)


@patch('tempfile.gettempdir', return_value="dummy_path/")
def test_temp_folder_generator(_):
    path = temp_folder_generator()
    assert path == "dummy_path/autokeras"
