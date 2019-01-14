import os
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from functools import reduce

from autokeras.constant import Constant
from autokeras.net_module import CnnModule
from autokeras.search import BayesianSearcher, train
from autokeras.utils import rand_temp_folder_generator, pickle_from_file, validate_xy, pickle_to_file, ensure_dir


class Supervised(ABC):
    """The base class for all supervised task.

    Attributes:
        verbose: A boolean value indicating the verbosity mode.
    """

    def __init__(self, verbose=False):
        """Initialize the instance.

        Args:
            verbose: A boolean of whether the search process will be printed to stdout.
        """
        self.verbose = verbose

    @abstractmethod
    def fit(self, x, y, time_limit=None):
        """Find the best neural architecture and train it.

        Based on the given dataset, the function will find the best neural architecture for it.
        The dataset is in numpy.ndarray format.
        So they training data should be passed through `x_train`, `y_train`.

        Args:
            x: A numpy.ndarray instance containing the training data or the training data combined with the
               validation data.
            y: A numpy.ndarray instance containing the label of the training data. or the label of the training data
               combined with the validation label.
            time_limit: The time limit for the search in seconds.
        """

    @abstractmethod
    def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=False):
        """Final training after found the best architecture.

        Args:
            x_train: A numpy.ndarray of training data.
            y_train: A numpy.ndarray of training targets.
            x_test: A numpy.ndarray of testing data.
            y_test: A numpy.ndarray of testing targets.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
            retrain: A boolean of whether reinitialize the weights of the model.
        """

    @abstractmethod
    def predict(self, x_test):
        """Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the results.
        """
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`."""
        pass


class DeepSupervised(Supervised):

    def __init__(self, verbose=False, path=None, resume=False, searcher_args=None,
                 search_type=BayesianSearcher):
        """Initialize the instance.

        The classifier will be loaded from the files in 'path' if parameter 'resume' is True.
        Otherwise it would create a new one.
        Args:
            verbose: A boolean of whether the search process will be printed to stdout.
            path: A string. The path to a directory, where the intermediate results are saved.
            resume: A boolean. If True, the classifier will continue to previous work saved in path.
                Otherwise, the classifier will start a new search.
            searcher_args: A dictionary containing the parameters for the searcher's __init__ function.
            search_type: A constant denoting the type of hyperparameter search algorithm that must be used.
        """
        super().__init__(verbose)

        if searcher_args is None:
            searcher_args = {}

        if path is None:
            path = rand_temp_folder_generator()

        self.path = path
        ensure_dir(path)
        if resume:
            classifier = pickle_from_file(os.path.join(self.path, 'classifier'))
            self.__dict__ = classifier.__dict__
            self.cnn = pickle_from_file(os.path.join(self.path, 'module'))
        else:
            self.y_encoder = None
            self.data_transformer = None
            self.verbose = verbose
            self.cnn = CnnModule(self.loss, self.metric, searcher_args, path, verbose, search_type)

    def fit(self, x, y, time_limit=None):
        validate_xy(x, y)
        y = self.transform_y(y)
        # Divide training data into training and testing data.
        validation_set_size = int(len(y) * Constant.VALIDATION_SET_SIZE)
        validation_set_size = min(validation_set_size, 500)
        validation_set_size = max(validation_set_size, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=validation_set_size,
                                                            random_state=42)
        self.init_transformer(x)
        # Transform x_train

        # Wrap the data into DataLoaders
        train_data = self.data_transformer.transform_train(x_train, y_train)
        test_data = self.data_transformer.transform_test(x_test, y_test)

        # Save the classifier
        pickle_to_file(self, os.path.join(self.path, 'classifier'))

        if time_limit is None:
            time_limit = 24 * 60 * 60

        self.cnn.fit(self.get_n_output_node(), x_train.shape, train_data, test_data, time_limit)

    def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=False):
        """Final training after found the best architecture.

        Args:
            x_train: A numpy.ndarray of training data.
            y_train: A numpy.ndarray of training targets.
            x_test: A numpy.ndarray of testing data.
            y_test: A numpy.ndarray of testing targets.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
            retrain: A boolean of whether reinitialize the weights of the model.
        """
        x_train = self.preprocess(x_train)
        x_test = self.preprocess(x_test)
        if trainer_args is None:
            trainer_args = {'max_no_improvement_num': 30}

        y_train = self.transform_y(y_train)
        y_test = self.transform_y(y_test)

        train_data = self.data_transformer.transform_train(x_train, y_train)
        test_data = self.data_transformer.transform_test(x_test, y_test)

        self.cnn.final_fit(train_data, test_data, trainer_args, retrain)

    @property
    @abstractmethod
    def metric(self):
        pass

    @property
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def get_n_output_node(self):
        pass

    def transform_y(self, y_train):
        return y_train

    def inverse_transform_y(self, output):
        return output

    @abstractmethod
    def init_transformer(self, x):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    def export_keras_model(self, model_file_name):
        """ Exports the best Keras model to the given filename. """
        self.cnn.best_model.produce_keras_model().save(model_file_name)

    def predict(self, x_test):
        """Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the results.
        """
        x_test = self.preprocess(x_test)
        test_loader = self.data_transformer.transform_test(x_test)
        return self.inverse_transform_y(self.cnn.predict(test_loader))

    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`."""
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_predict, y_test)


class PortableClass(ABC):
    def __init__(self, graph, verbose=False):
        self.graph = graph
        self.verbose = verbose

    @abstractmethod
    def fit(self, **kwargs):
        """further training of the model (graph).
        """
        pass

    @abstractmethod
    def predict(self, x_test):
        """Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the results.
        """
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`."""
        pass


class PortableDeepSupervised(PortableClass):
    def __init__(self, graph, y_encoder, data_transformer, verbose=False, path=None):
        """Initialize the instance.

        Args:
            graph: The graph form of the learned model.
            y_encoder: The encoder of the label. See example as OneHotEncoder
            data_transformer: A transformer class to process the data. See example as ImageDataTransformer.
            verbose: A boolean of whether the search process will be printed to stdout.
            path: A string. The path to a directory, where the intermediate results are saved.
        """
        super(PortableDeepSupervised, self).__init__(graph, verbose)
        self.y_encoder = y_encoder
        self.data_transformer = data_transformer
        if path is None:
            path = rand_temp_folder_generator()
        self.path = path

    def fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=False):
        """further training of the model (graph).

        Args:
            x_train: A numpy.ndarray of training data.
            y_train: A numpy.ndarray of training targets.
            x_test: A numpy.ndarray of testing data.
            y_test: A numpy.ndarray of testing targets.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
            retrain: A boolean of whether reinitialize the weights of the model.
        """
        x_train = self.preprocess(x_train)
        x_test = self.preprocess(x_test)
        if trainer_args is None:
            trainer_args = {'max_no_improvement_num': 30}

        y_train = self.transform_y(y_train)
        y_test = self.transform_y(y_test)

        train_data = self.data_transformer.transform_train(x_train, y_train)
        test_data = self.data_transformer.transform_test(x_test, y_test)

        if retrain:
            self.graph.weighted = False
        _, _1, self.graph = train(None, self.graph, train_data, test_data, trainer_args,
                                  self.metric, self.loss, self.verbose, self.path)

    @property
    @abstractmethod
    def metric(self):
        pass

    @property
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    @abstractmethod
    def transform_y(self, y):
        pass

    @abstractmethod
    def inverse_transform_y(self, output):
        pass

    def predict(self, x_test):
        """Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the results.
        """
        if Constant.LIMIT_MEMORY:
            pass

        x_test = self.preprocess(x_test)
        test_loader = self.data_transformer.transform_test(x_test)
        model = self.graph.produce_model()
        model.eval()

        outputs = []
        with torch.no_grad():
            for index, inputs in enumerate(test_loader):
                outputs.append(model(inputs).numpy())
        output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
        return self.inverse_transform_y(output)

    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`."""
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_predict, y_test)
