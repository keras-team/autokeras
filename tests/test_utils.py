from autokeras.generator import DefaultClassifierGenerator
from autokeras.metric import Accuracy
from autokeras.utils import *

from tests.common import get_processed_data


def test_model_trainer():
    model = DefaultClassifierGenerator(3, (28, 28, 3)).generate().produce_model()
    train_data, test_data = get_processed_data()
    ModelTrainer(model, train_data, test_data, Accuracy, False).train_model(max_iter_num=3)
