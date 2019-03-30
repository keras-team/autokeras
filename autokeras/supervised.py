import os
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

from autokeras.backend import Backend
from autokeras.constant import Constant
from autokeras.net_module import CnnModule
from autokeras.search import BayesianSearcher, train
from autokeras.utils import rand_temp_folder_generator, pickle_from_file, validate_xy, pickle_to_file, ensure_dir


class Supervised(ABC):
    """The base class for all supervised tasks.

    Attributes:
        verbose: A boolean value indicating the verbosity mode.
    """

    def __init__(self, verbose=False):
        """Initialize the instance of the class.

        Args:
            verbose: A boolean of whether the search process will be printed to stdout. (optional, default = False)
        """
        self.verbose = verbose

    @abstractmethod
    def fit(self, x_train, y_train, time_limit=None):
        """Find the best neural architecture for classifying the training data and train it.

        Based on the given dataset, the function will find the best neural architecture for it.
        The dataset must be in numpy.ndarray format.
        So the training data should be passed through `x_train`, `y_train`.

        Args:
            x_train: A numpy.ndarray instance containing the training data or the training data combined with the
               validation data.
            y_train: A numpy.ndarray instance containing the labels of the training data. or the label of the training data
               combined with the validation label.
            time_limit: The time limit for the search in seconds.
            
        Effects:
            Trains a model that fits the data using the best neural architecture
        """
        pass

    @abstractmethod
    def predict(self, x_test):
        """Return the results for the testing data predicted by the best neural architecture.
        
        Dependent on the results of the fit() function.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the predicted classes for x_test.
        """
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`.
        
        Args:
            x_test: An instance of numpy.ndarray containing the testing data
            y_test: An instance of numpy.ndarray containing the labels of the testing data
            
        Returns:
            A float value of the accuracy of the predictions given the labels for the testing data
        """
        pass


class SearchSupervised(Supervised):
    """The base class for all supervised tasks using neural architecture search.
    
    Inherits from Supervised class.
    
    Attributes:
        verbose: A boolean value indicating the verbosity mode.
    """

    @abstractmethod
    def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=False):
        """Final training after finding the best neural architecture.

        Args:
            x_train: A numpy.ndarray of training data.
            y_train: A numpy.ndarray of training targets.
            x_test: A numpy.ndarray of testing data.
            y_test: A numpy.ndarray of testing targets.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor. (optional, default = None)
            retrain: A boolean of whether reinitialize the weights of the model. (optional, default = False)
        """
        pass


class DeepTaskSupervised(SearchSupervised):
    """
    Inherits from SearchSupervised class.
    
    Attributes:
        verbose: A boolean value indicating the verbosity mode. (optional, default = False)
        path: A string indicating the path to a directory where the intermediate results are saved. (optional, default = None)
        resume: A boolean. If True, the classifier will continue to previous work saved in path.
            Otherwise, the classifier will start a new search. (optional, default = False)
        searcher_args: A dictionary containing the parameters for the searcher's __init__ function. (optional, default = None)
        search_type: A constant denoting the type of hyperparameter search algorithm that must be used. (optional, default = BayesianSearcher)
    """

    def __init__(self, verbose=False, path=None, resume=False, searcher_args=None,
                 search_type=BayesianSearcher):
        """Initialize the instance of a DeepTaskSupervised class.

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
        """Find the best neural architecture for classifying the training data and train it.

        Based on the given dataset, the function will find the best neural architecture for it.
        The dataset must be in numpy.ndarray format.
        The training and validation data should be passed through `x`, `y`. This method will automatically split
        the training and validation data into training and validation sets.

        Args:
            x: A numpy.ndarray instance containing the training data or the training data combined with the
               validation data.
            y: A numpy.ndarray instance containing the labels of the training data. or the label of the training data
               combined with the validation label.
            time_limit: The time limit for the search in seconds. (optional, default = None, which turns into 24 hours in method)
            
        Effects:
            Trains a model that fits the data using the best neural architecture
        """
        validate_xy(x, y)
        y = self.transform_y(y)
        # Divide training data into training and validation data.
        validation_set_size = int(len(y) * Constant.VALIDATION_SET_SIZE)
        validation_set_size = min(validation_set_size, 500)
        validation_set_size = max(validation_set_size, 1)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y,
                                                              test_size=validation_set_size,
                                                              random_state=42)
        self.init_transformer(x)
        # Transform x_train

        # Wrap the data into DataLoaders
        train_data = self.data_transformer.transform_train(x_train, y_train)
        valid_data = self.data_transformer.transform_test(x_valid, y_valid)

        # Save the classifier
        pickle_to_file(self, os.path.join(self.path, 'classifier'))

        if time_limit is None:
            time_limit = 24 * 60 * 60

        self.cnn.fit(self.get_n_output_node(), x_train.shape, train_data, valid_data, time_limit)

    def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=False):
        """Final training after found the best architecture.

        Args:
            x_train: A numpy.ndarray of training data.
            y_train: A numpy.ndarray of training targets.
            x_test: A numpy.ndarray of testing data.
            y_test: A numpy.ndarray of testing targets.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
            retrain: A boolean indicating whether or not to reinitialize the weights of the model.
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

    @staticmethod
    def transform_y(y_train):
        return y_train

    @staticmethod
    def inverse_transform_y(output):
        return output

    @abstractmethod
    def init_transformer(self, x):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    def export_keras_model(self, model_file_name):
        """Exports the best Keras model to the given filename.
        
        Args:
            model_file_name: A string of the filename to which the best model will be exported
        
        Effects:
            Save the architecture, weights, and optimizer state of the best model
        """
        self.cnn.best_model.produce_keras_model().save(model_file_name)

    def predict(self, x_test):
        """Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the predictions for the testing data.
        """
        x_test = self.preprocess(x_test)
        test_loader = self.data_transformer.transform_test(x_test)
        return self.inverse_transform_y(self.cnn.predict(test_loader))

    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`.
        
        Predict the labels for the testing data.
        Calculate the accuracy metric between the predicted and actual labels of the testing data.
        
        Args:
            x_test: An instance of numpy.ndarray containing the testing data
            y_test: An instance of numpy.ndarray containing the labels of the testing data
            
        Returns:
            A float value of the accuracy of the predictions given the labels for the testing data
        """
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_predict, y_test)


class SingleModelSupervised(Supervised):
    """The base class for all supervised tasks that do not use neural architecture search.
    
    Inherits from Supervised class.
    
    Attributes:
        verbose: A boolean value indicating the verbosity mode.
        path: A string value indicating the path to the directory where the intermediate model results 
              are stored
        graph: The graph form of the learned model.
        data_transformer: A transformer class to process the data. (See example `ImageDataTransformer`.)
        verbose: A boolean of whether the search process will be printed to stdout.
    """

    def __init__(self, verbose=False, path=None):
        """Initialize the instance of the SingleModelSupervised class.

        Args:
            verbose: A boolean of whether the search process will be printed to stdout. (optional, default = False)
            path: A string. The path to a directory, where the intermediate results are saved. (optional, default = None)
        """
        super().__init__(verbose)
        if path is None:
            path = rand_temp_folder_generator()
        self.path = path
        self.graph = None
        self.data_transformer = None

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
        """Return the predicted labels for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the predicted labels for the testing data.
        """
        if Constant.LIMIT_MEMORY:
            pass

        x_test = self.preprocess(x_test)
        test_loader = self.data_transformer.transform_test(x_test)
        model = self.graph.produce_model()
        model.eval()

        output = Backend.predict(model, test_loader)
        return self.inverse_transform_y(output)

    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`.
        
        Predict the labels for the testing data.
        Calculate the accuracy metric between the predicted and actual labels of the testing data.
        
        Args:
            x_test: An instance of numpy.ndarray containing the testing data
            y_test: An instance of numpy.ndarray containing the labels of the testing data
            
        Returns:
            A float value of the accuracy of the predictions given the labels for the testing data
        """
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_predict, y_test)

    def save(self, model_path):
        """Exports the Keras model to the given filename.
        
        Args:
            model_path: A string of the path to which the model will be saved
        
        Effects:
            Save the architecture, weights, and optimizer state of the best model
        """
        self.graph.produce_keras_model().save(model_path)


class PortableDeepSupervised(SingleModelSupervised, ABC):
    """The basis class for exported keras model
    
    Inheirits from SingleModelSupervised class and abc module.
    
    Attributes:
        graph: The graph form of the learned model.
        y_encoder: The encoder of the label. (See example `OneHotEncoder`.)
        data_transformer: A transformer class to process the data. (See example `ImageDataTransformer`.)
        verbose: A boolean of whether the search process will be printed to stdout.
        path: A string value indicating the path to the directory where the intermediate model results
              are stored
    """

    def __init__(self, graph, y_encoder, data_transformer, verbose=False, path=None):
        """Initialize the instance of the PortableDeepSupervised class.

        Args:
            graph: The graph form of the learned model.
            y_encoder: The encoder of the label. See example as OneHotEncoder
            data_transformer: A transformer class to process the data. See example as ImageDataTransformer.
            verbose: A boolean of whether the search process will be printed to stdout.
            path: A string. The path to a directory, where the intermediate results are saved.
        """
        super().__init__(verbose, path)
        self.graph = graph
        self.y_encoder = y_encoder
        self.data_transformer = data_transformer

    def fit(self, x, y, trainer_args=None, retrain=False):
        """Trains the model on the given dataset.

        Args:
            x: A numpy.ndarray instance containing the training data or the training data combined with the
               validation data.
            y: A numpy.ndarray instance containing the label of the training data. or the label of the training data
               combined with the validation label.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
            retrain: A boolean of whether reinitialize the weights of the model.
        """
        x = self.preprocess(x)
        # Divide training data into training and testing data.
        validation_set_size = int(len(y) * Constant.VALIDATION_SET_SIZE)
        validation_set_size = min(validation_set_size, 500)
        validation_set_size = max(validation_set_size, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=validation_set_size,
                                                            random_state=42)
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
