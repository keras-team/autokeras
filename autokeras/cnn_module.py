import os
import pickle
import time

from autokeras.constant import Constant
from autokeras.search import Searcher, train
from autokeras.utils import pickle_from_file




class CnnModule(object):
    def __init__(self, loss, metric, searcher_args, path, verbose=False):
        self.searcher_args = searcher_args
        self.searcher = None
        self.path = path
        self.verbose = verbose
        self.loss = loss
        self.metric = metric

    def fit(self, n_output_node, input_shape, train_data, test_data, time_limit=24 * 60 * 60):
        """ Search the best CnnModule.

        Args:
            n_output_node: A integer value represent the number of output node in the final layer.
            input_shape: A tuple to express the shape of every train entry. For example,
                MNIST dataset would be (28,28,1)
            train_data: A PyTorch DataLoader instance represents the training data
            test_data: A PyTorch DataLoader instance represents the testing data
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
            self.searcher_args['verbose'] = self.verbose
            searcher = Searcher(**self.searcher_args)
            self._save_searcher(searcher)
            self.searcher = True

        start_time = time.time()
        time_remain = time_limit
        try:
            while time_remain > 0:
                searcher = pickle_from_file(os.path.join(self.path, 'searcher'))
                searcher.search(train_data, test_data, int(time_remain))
                if len(self._load_searcher().history) >= Constant.MAX_MODEL_NUM:
                    break
                time_elapsed = time.time() - start_time
                time_remain = time_limit - time_elapsed
            # if no search executed during the time_limit, then raise an error
            if time_remain <= 0:
                raise TimeoutError
        except TimeoutError:
            if len(self._load_searcher().history) == 0:
                raise TimeoutError("Search Time too short. No model was found during the search time.")
            elif self.verbose:
                print('Time is out.')

    def final_fit(self, train_data, test_data, trainer_args=None, retrain=False):
        """Final training after found the best architecture.

        Args:
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
            retrain: A boolean of whether reinitialize the weights of the model.
            train_data: A DataLoader instance representing the training data
            test_data: A DataLoader instance representing the testing data

        """
        searcher = self._load_searcher()
        graph = searcher.load_best_model()

        if retrain:
            graph.weighted = False
        _, _1, graph = train((graph,
                              train_data,
                              test_data,
                              trainer_args,
                              self.metric,
                              self.loss,
                              self.verbose,
                              self.path))
        searcher.replace_model(graph, searcher.get_best_model_id())

    @property
    def best_model(self):
        return self._load_searcher().load_best_model()

    def _save_searcher(self, searcher):
        pickle.dump(searcher, open(os.path.join(self.path, 'searcher'), 'wb'))

    def _load_searcher(self):
        return pickle_from_file(os.path.join(self.path, 'searcher'))
