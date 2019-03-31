import os
import time

from autokeras.backend import Backend
from autokeras.constant import Constant
from autokeras.search import BayesianSearcher, train

from autokeras.utils import pickle_to_file, rand_temp_folder_generator, ensure_dir
from autokeras.nn.generator import CnnGenerator, MlpGenerator, ResNetGenerator, DenseNetGenerator


class NetworkModule:
    """ Class to create a network module.

    Attributes:
        loss: A function taking two parameters, the predictions and the ground truth.
        metric: An instance of the Metric subclasses.
        searcher_args: A dictionary containing the parameters for the searcher's __init__ function.
        searcher: An instance of the Searcher class.
        path: A string. The path to the directory to save the searcher.
        verbose: A boolean. Setting it to true prints to stdout.
        generators: A list of instances of the NetworkGenerator class or its subclasses.
        search_type: A constant denoting the type of hyperparameter search algorithm that must be used.
    """

    def __init__(self, loss, metric, searcher_args=None, path=None, verbose=False, search_type=BayesianSearcher, skip_conn=True):
        self.searcher_args = searcher_args if searcher_args is not None else {}
        self.searcher = None
        self.path = path if path is not None else rand_temp_folder_generator()
        ensure_dir(self.path)
        if verbose:
            print('Saving Directory:', self.path)
        self.verbose = verbose
        self.loss = loss
        self.metric = metric
        self.generators = []
        self.search_type = search_type
        self.skip_conn = skip_conn

    def fit(self, n_output_node, input_shape, train_data, test_data, time_limit=24 * 60 * 60):
        """ Search the best network.

        Args:
            n_output_node: A integer value represent the number of output node in the final layer.
            input_shape: A tuple to express the shape of every train entry. For example,
                MNIST dataset would be (28,28,1).
            train_data: A PyTorch DataLoader instance representing the training data.
            test_data: A PyTorch DataLoader instance representing the testing data.
            time_limit: A integer value represents the time limit on searching for models.
        """
        # Create the searcher and save on disk

        if not self.searcher:
            input_shape = input_shape[1:]
            self.searcher_args['n_output_node'] = n_output_node
            self.searcher_args['input_shape'] = input_shape
            self.searcher_args['path'] = self.path
            self.searcher_args['metric'] = self.metric
            self.searcher_args['loss'] = self.loss
            self.searcher_args['generators'] = self.generators
            self.searcher_args['verbose'] = self.verbose
            pickle_to_file(self, os.path.join(self.path, 'module'))
            self.searcher = self.search_type(**self.searcher_args, skip_conn=self.skip_conn)

        start_time = time.time()
        time_remain = time_limit
        try:
            while time_remain > 0:
                self.searcher.search(train_data, test_data, int(time_remain))
                pickle_to_file(self, os.path.join(self.path, 'module'))
                if len(self.searcher.history) >= Constant.MAX_MODEL_NUM:
                    break
                time_elapsed = time.time() - start_time
                time_remain = time_limit - time_elapsed
            # if no search executed during the time_limit, then raise an error
            if time_remain <= 0:
                raise TimeoutError
        except TimeoutError:
            if len(self.searcher.history) == 0:
                raise TimeoutError("Search Time too short. No model was found during the search time.")
            elif self.verbose:
                print('Time is out.')

    def final_fit(self, train_data, test_data, trainer_args=None, retrain=False):
        """Final training after found the best architecture.

        Args:
            train_data: A DataLoader instance representing the training data.
            test_data: A DataLoader instance representing the testing data.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
            retrain: A boolean of whether reinitialize the weights of the model.
        """
        graph = self.searcher.load_best_model()

        if retrain:
            graph.weighted = False
        _, _1, graph = train(None, graph,
                             train_data,
                             test_data,
                             trainer_args,
                             self.metric,
                             self.loss,
                             self.verbose,
                             self.path)
        self.searcher.replace_model(graph, self.searcher.get_best_model_id())
        pickle_to_file(self, os.path.join(self.path, 'module'))

    @property
    def best_model(self):
        return self.searcher.load_best_model()

    def predict(self, test_loader):
        model = self.best_model.produce_model()
        model.eval()

        return Backend.predict(model, test_loader)


class CnnModule(NetworkModule):
    """ Class to create a CNN module."""

    def __init__(self, loss, metric, searcher_args=None, path=None, verbose=False,
                 search_type=BayesianSearcher):
        super(CnnModule, self).__init__(loss, metric, searcher_args, path, verbose, search_type)
        self.generators.append(CnnGenerator)
        self.generators.append(ResNetGenerator)
        self.generators.append(DenseNetGenerator)


class MlpModule(NetworkModule):
    """ Class to create an MLP module."""

    def __init__(self, loss, metric, searcher_args=None, path=None, verbose=False):
        super(MlpModule, self).__init__(loss, metric, searcher_args, path, verbose, skip_conn=False)
        self.generators.extend([MlpGenerator] * 2)
