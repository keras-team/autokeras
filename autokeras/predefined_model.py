import os
import numpy as np
import torch
from abc import ABC, abstractmethod
from functools import reduce
from sklearn.model_selection import train_test_split

from autokeras.utils import rand_temp_folder_generator, validate_xy
from autokeras.nn.metric import Accuracy
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.generator import ResNetGenerator, DenseNetGenerator
from autokeras.search import train
from autokeras.constant import Constant
from autokeras.preprocessor import ImageDataTransformer, OneHotEncoder

class PredefinedModel(ABC):
    """The base class for the predefined model without architecture search

    Attributes:
        y_encoder: Label encoder, used in transform_y or inverse_transform_y for encode the label. For example,
                   if one hot encoder needed, y_encoder can be OneHotEncoder.
        data_transformer_class: A transformer class to process the data. See example as ImageDataTransformer.
        data_transformer: A instance of data_transformer_class.
        verbose: A boolean of whether the search process will be printed to stdout.
        path: A string. The path to a directory, where the intermediate results are saved.
    """
    def __init__(self, y_encoder=OneHotEncoder, data_transformer_class=ImageDataTransformer,
                 verbose=False,
                 path=None):
        self.graph = None
        self.generator = None
        self.loss = classification_loss
        self.metric = Accuracy
        self.y_encoder = y_encoder()
        self.data_transformer_class = data_transformer_class
        self.data_transformer = None
        self.verbose = verbose
        if path is None:
            path = rand_temp_folder_generator()
        self.path = path

    @abstractmethod
    def _init_generator(self, n_output_node, input_shape):
        """Initialize the generator to generate the model architecture.

        Args:
            n_output_node:  A integer value represent the number of output node in the final layer.
            input_shape: A tuple to express the shape of every train entry.
        """
        pass

    def compile(self, loss=classification_loss, metric=Accuracy):
        """Configures the model for training.

        Args:
            loss: The loss function to train the model. See example as classification_loss.
            metric: The metric to be evaluted by the model during training and testing.
                    See example as Accuracy.
        """
        self.loss = loss
        self.metric = metric

    def fit(self, x, y, trainer_args=None):
        """Trains the model on the dataset given.

        Args:
            x: A numpy.ndarray instance containing the training data or the training data combined with the
               validation data.
            y: A numpy.ndarray instance containing the label of the training data. or the label of the training data
               combined with the validation label.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
        """
        validate_xy(x, y)
        self.y_encoder.fit(y)
        y = self.y_encoder.transform(y)
        # Divide training data into training and testing data.
        validation_set_size = int(len(y) * Constant.VALIDATION_SET_SIZE)
        validation_set_size = min(validation_set_size, 500)
        validation_set_size = max(validation_set_size, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=validation_set_size,
                                                            random_state=42)

        #initialize data_transformer
        self.data_transformer = self.data_transformer_class(x_train)
        # Wrap the data into DataLoaders
        train_loader = self.data_transformer.transform_train(x_train, y_train)
        test_loader = self.data_transformer.transform_test(x_test, y_test)

        self.generator = self._init_generator(self.y_encoder.n_classes, x_train.shape[1:])
        graph = self.generator.generate()

        if trainer_args is None:
            trainer_args = {'max_no_improvement_num': 30}
        _, _1, self.graph = train(None, graph, train_loader, test_loader,
                                  trainer_args, self.metric, self.loss,
                                  self.verbose, self.path)

    def predict(self, x_test):
        """Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the results.
        """
        test_loader = self.data_transformer.transform_test(x_test)
        model = self.graph.produce_model()
        model.eval()

        outputs = []
        with torch.no_grad():
            for index, inputs in enumerate(test_loader):
                outputs.append(model(inputs).numpy())
        output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
        return self.y_encoder.inverse_transform(output)

    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`.
        """
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_predict, y_test)

    def save(self, model_path):
        """Save the model as keras format.

        Args:
            model_path: the path to save model.
        """
        self.graph.produce_keras_model().save(model_path)


class PredefinedResnet(PredefinedModel):
    def _init_generator(self, n_output_node, input_shape):
        return ResNetGenerator(n_output_node, input_shape)


class PredefinedDensenet(PredefinedModel):
    def _init_generator(self, n_output_node, input_shape):
        return DenseNetGenerator(n_output_node, input_shape)