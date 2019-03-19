from autokeras.backend import Backend, tensorflow, torch
from autokeras.nn.generator import CnnGenerator
from autokeras.nn.metric import Accuracy, MSE
from autokeras.backend.tensorflow.model_trainer import ModelTrainer
from tests.common import get_classification_data_loaders, \
    get_regression_data_loaders, clean_dir, TEST_TEMP_DIR
import pytest


# def test_model_trainer_classification():
#     Backend.backend = tensorflow
#     model = CnnGenerator(3, (28, 28, 3)).generate().produce_model()
#     train_data, test_data = get_classification_data_loaders()
#     ModelTrainer(model,
#                  train_data=train_data,
#                  test_data=test_data,
#                  metric=Accuracy,
#                  loss_function=Backend.classification_loss,
#                  verbose=True,
#                  path=TEST_TEMP_DIR).train_model(max_iter_num=3)
#     Backend.backend = torch
#     clean_dir(TEST_TEMP_DIR)
#
#
# def test_model_trainer_regression():
#     Backend.backend = tensorflow
#     model = CnnGenerator(1, (28, 28, 3)).generate().produce_model()
#     train_data, test_data = get_regression_data_loaders()
#     ModelTrainer(model,
#                  train_data=train_data,
#                  test_data=test_data,
#                  metric=MSE,
#                  loss_function=Backend.regression_loss,
#                  verbose=False,
#                  path=TEST_TEMP_DIR).train_model(max_iter_num=3)
#     Backend.backend = torch
#     clean_dir(TEST_TEMP_DIR)


# def test_produce_keras_model():
#     for graph in [get_conv_dense_model(),
#                   get_add_skip_model(),
#                   get_pooling_model(),
#                   get_concat_skip_model()]:
#         model = graph.produce_keras_model()
#         assert isinstance(model, models.Model)
#
#
# def test_keras_model():
#     for graph in [get_conv_dense_model(),
#                   get_add_skip_model(),
#                   get_pooling_model(),
#                   get_concat_skip_model()]:
#         keras_model = model.KerasModel(graph)
#         keras_model.set_weight_to_graph()
#         assert isinstance(keras_model, model.KerasModel)