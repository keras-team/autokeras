import os
import numpy as np
import torch
from abc import ABC, abstractmethod
from functools import reduce

from autokeras.utils import rand_temp_folder_generator, pickle_to_file
from autokeras.nn.metric import Accuracy
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.generator import ResNetGenerator, DenseNetGenerator
from autokeras.search import train

class PredefinedModel(ABC):
    def __init__(self, n_output_node, input_shape, inverse_transform_y_method, verbose=False, path=None):
        self.generator = self.init_generator(n_output_node, input_shape)
        self.graph = None
        self.loss = classification_loss
        self.metric = Accuracy
        self.inverse_transform_y_method = inverse_transform_y_method
        self.verbose = verbose
        if path is None:
            path = rand_temp_folder_generator()
        self.path = path

    @abstractmethod
    def init_generator(self, n_output_node, input_shape):
        pass

    def compile(self, loss=classification_loss, metric=Accuracy):
        self.loss = loss
        self.metric = metric

    def fit(self, train_loader, validation_loader, trainer_args=None):
        graph = self.generator.generate()

        if trainer_args is None:
            trainer_args = {'max_no_improvement_num': 30}
        _, _1, self.graph = train(None, graph, train_loader, validation_loader,
                                  trainer_args, self.metric, self.loss,
                                  self.verbose, self.path)

    def predict(self, test_loader):
        model = self.graph.produce_model()
        model.eval()

        outputs = []
        with torch.no_grad():
            for index, inputs in enumerate(test_loader):
                outputs.append(model(inputs).numpy())
        output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
        return self.inverse_transform_y_method(output)

    def evaluate(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_predict, y_test)

    def save(self, model_path):
        self.graph.produce_keras_model().save(model_path)


class PredefinedResnet(PredefinedModel):
    def init_generator(self, n_output_node, input_shape):
        return ResNetGenerator(n_output_node, input_shape)


class PredefinedDensenet(PredefinedModel):
    def init_generator(self, n_output_node, input_shape):
        return DenseNetGenerator(n_output_node, input_shape)