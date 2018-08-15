from unittest.mock import patch

from autokeras.generator import DefaultClassifierGenerator
from autokeras.metric import Accuracy
from autokeras.utils import ModelTrainer, temp_folder_generator
from tests.common import get_processed_data


def test_model_trainer():
    model = DefaultClassifierGenerator(3, (28, 28, 3)).generate().produce_model()
    train_data, test_data = get_processed_data()
    ModelTrainer(model, train_data, test_data, Accuracy, False).train_model(max_iter_num=3)


@patch('tempfile.gettempdir', return_value="dummy_path/")
def test_temp_folder_generator(_):
    path = temp_folder_generator()
    assert path == "dummy_path/autokeras"
