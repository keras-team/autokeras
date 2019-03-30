import logging
import os
import queue
import re
import sys
import time
import torch
import torch.multiprocessing as mp

from abc import ABC, abstractmethod
from datetime import datetime

from autokeras.backend import Backend
from autokeras.bayesian import BayesianOptimizer
from autokeras.constant import Constant
from autokeras.utils import pickle_to_file, pickle_from_file, verbose_print, get_system


class Searcher(ABC):
    """The base class to search for neural architectures.

    This class generate new architectures, call the trainer to train it, and update the optimizer.

    Attributes:
        n_classes: Number of classes in the target classification task.
        input_shape: Arbitrary, although all dimensions in the input shaped must be fixed.
            Use the keyword argument `input_shape` (tuple of integers, does not include the batch axis)
            when using this layer as the first layer in a model.
        verbose: Verbosity mode.
        history: A list that stores the performance of model. Each element in it is a dictionary of 'model_id',
            'loss', and 'metric_value'.
        neighbour_history: A list that stores the performance of neighbor of the best model.
            Each element in it is a dictionary of 'model_id', 'loss', and 'metric_value'.
        path: A string. The path to the directory for saving the searcher.
        metric: An instance of the Metric subclasses.
        loss: A function taking two parameters, the predictions and the ground truth.
        generators: A list of generators used to initialize the search.
        model_count: An integer. the total number of neural networks in the current searcher.
        descriptors: A dictionary of all the neural network architectures searched.
        trainer_args: A dictionary. The params for the constructor of ModelTrainer.
        default_model_len: An integer. Number of convolutional layers in the initial architecture.
        default_model_width: An integer. The number of filters in each layer in the initial architecture.
        training_queue: A list of the generated architectures to be trained.
        x_queue: A list of trained architectures not updated to the gpr.
        y_queue: A list of trained architecture performances not updated to the gpr.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose,
                 trainer_args=None,
                 default_model_len=None,
                 default_model_width=None,
                 skip_conn=True):
        """Initialize the Searcher.

        Args:
            n_output_node: An integer, the number of classes.
            input_shape: A tuple. e.g. (28, 28, 1).
            path: A string. The path to the directory to save the searcher.
            metric: An instance of the Metric subclasses.
            loss: A function taking two parameters, the predictions and the ground truth.
            generators: A list of generators used to initialize the search.
            verbose: A boolean. Whether to output the intermediate information to stdout.
            trainer_args: A dictionary. The params for the constructor of ModelTrainer.
            default_model_len: An integer. Number of convolutional layers in the initial architecture.
            default_model_width: An integer. The number of filters in each layer in the initial architecture.
        """
        if trainer_args is None:
            trainer_args = {}
        self.n_classes = n_output_node
        self.input_shape = input_shape
        self.verbose = verbose
        self.history = []
        self.neighbour_history = []
        self.path = path
        self.metric = metric
        self.loss = loss
        self.generators = generators
        self.model_count = 0
        self.descriptors = []
        self.trainer_args = trainer_args
        self.default_model_len = default_model_len if default_model_len is not None else Constant.MODEL_LEN
        self.default_model_width = default_model_width if default_model_width is not None else Constant.MODEL_WIDTH
        self.skip_conn = skip_conn

        if 'max_iter_num' not in self.trainer_args:
            self.trainer_args['max_iter_num'] = Constant.SEARCH_MAX_ITER

        self.training_queue = []
        self.x_queue = []
        self.y_queue = []

        logging.basicConfig(filename=os.path.join(self.path, datetime.now().strftime('run_%d_%m_%Y_%H_%M.log')),
                            format='%(asctime)s - %(filename)s - %(message)s', level=logging.DEBUG)

        self._timeout = None

    def load_model_by_id(self, model_id):
        return pickle_from_file(os.path.join(self.path, str(model_id) + '.graph'))

    def load_best_model(self):
        return self.load_model_by_id(self.get_best_model_id())

    def get_metric_value_by_id(self, model_id):
        for item in self.history:
            if item['model_id'] == model_id:
                return item['metric_value']
        return None

    def get_best_model_id(self):
        if self.metric.higher_better():
            return max(self.history, key=lambda x: x['metric_value'])['model_id']
        return min(self.history, key=lambda x: x['metric_value'])['model_id']

    def replace_model(self, graph, model_id):
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.graph'))

    def init_search(self):
        """Call the generators to generate the initial architectures for the search."""
        if self.verbose:
            print('\nInitializing search.')
        for generator in self.generators:
            graph = generator(self.n_classes, self.input_shape). \
                generate(self.default_model_len, self.default_model_width)
            model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((graph, -1, model_id))
            self.descriptors.append(graph.extract_descriptor())

        if self.verbose:
            print('Initialization finished.')

    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        """Run the search loop of training, generating and updating once.

        The function will run the training and generate in parallel.
        Then it will update the controller.
        The training is just pop out a graph from the training_queue and train it.
        The generate will call the self.generate function.
        The update will call the self.update function.

        Args:
            train_data: An instance of DataLoader.
            test_data: An instance of Dataloader.
            timeout: An integer, time limit in seconds.
        """
        torch.cuda.empty_cache()
        if not self.history:
            self.init_search()

        self._timeout = time.time() + timeout if timeout is not None else sys.maxsize
        self.trainer_args['timeout'] = timeout
        # Start the new process for training.
        graph, other_info, model_id = self.training_queue.pop(0)
        if self.verbose:
            print('\n')
            print('+' + '-' * 46 + '+')
            print('|' + 'Training model {}'.format(model_id).center(46) + '|')
            print('+' + '-' * 46 + '+')
        # Temporary solution to support GOOGLE Colab
        if get_system() == Constant.SYS_GOOGLE_COLAB:
            # When using Google Colab, use single process for searching and training.
            self.sp_search(graph, other_info, model_id, train_data, test_data)
        else:
            # Use two processes
            self.mp_search(graph, other_info, model_id, train_data, test_data)

    def mp_search(self, graph, other_info, model_id, train_data, test_data):
        ctx = mp.get_context()
        q = ctx.Queue()
        p = ctx.Process(target=train, args=(q, graph, train_data, test_data, self.trainer_args,
                                            self.metric, self.loss, self.verbose, self.path))
        try:
            p.start()
            search_results = self._search_common(q)
            metric_value, loss, graph = q.get(block=True)
            if time.time() >= self._timeout:
                raise TimeoutError
            if self.verbose and search_results:
                for (generated_graph, generated_other_info, new_model_id) in search_results:
                    verbose_print(generated_other_info, generated_graph, new_model_id)

            if metric_value is not None:
                self.add_model(metric_value, loss, graph, model_id)
                self.update(other_info, model_id, graph, metric_value)

        except (TimeoutError, queue.Empty) as e:
            raise TimeoutError from e
        finally:
            # terminate and join the subprocess to prevent any resource leak
            p.terminate()
            p.join()

    def sp_search(self, graph, other_info, model_id, train_data, test_data):
        try:
            metric_value, loss, graph = train(None, graph, train_data, test_data, self.trainer_args,
                                              self.metric, self.loss, self.verbose, self.path)
            # Do the search in current thread.
            search_results = self._search_common()
            if self.verbose and search_results:
                for (generated_graph, generated_other_info, new_model_id) in search_results:
                    verbose_print(generated_other_info, generated_graph, new_model_id)

            if metric_value is not None:
                self.add_model(metric_value, loss, graph, model_id)
                self.update(other_info, model_id, graph, metric_value)

        except TimeoutError as e:
            raise TimeoutError from e

    def _search_common(self, mp_queue=None):
        search_results = []
        if not self.training_queue:
            results = self.generate(mp_queue)
            for (generated_graph, generated_other_info) in results:
                new_model_id = self.model_count
                self.model_count += 1
                self.training_queue.append((generated_graph, generated_other_info, new_model_id))
                self.descriptors.append(generated_graph.extract_descriptor())
                search_results.append((generated_graph, generated_other_info, new_model_id))
            self.neighbour_history = []

        return search_results

    @abstractmethod
    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.

        Returns:
            list of 2-element tuples: generated_graph and other_info,
            generated_graph: An instance of Graph.
            other_info: Anything to be saved in the training queue together
            with the architecture.
        """
        pass

    @abstractmethod
    def update(self, other_info, model_id, graph, metric_value):
        """ Update the controller with evaluation result of a neural architecture.

        Args:
            other_info: Anything. In the case of default bayesian searcher, it is the father ID in the search tree.
            model_id: An integer.
            graph: An instance of Graph. The trained neural architecture.
            metric_value: The final evaluated metric value.
        """
        pass

    def add_model(self, metric_value, loss, graph, model_id):
        """Append the information of evaluated architecture to history."""
        if self.verbose:
            print('\nSaving model.')

        graph.clear_operation_history()
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.graph'))

        ret = {'model_id': model_id, 'loss': loss, 'metric_value': metric_value}
        self.neighbour_history.append(ret)
        self.history.append(ret)

        # Update best_model text file
        if model_id == self.get_best_model_id():
            file = open(os.path.join(self.path, 'best_model.txt'), 'w')
            file.write('best model: ' + str(model_id))
            file.close()

        if self.verbose:
            idx = ['model_id', 'loss', 'metric_value']
            header = ['Model ID', 'Loss', 'Metric Value']
            line = '|'.join(x.center(24) for x in header)
            print('+' + '-' * len(line) + '+')
            print('|' + line + '|')

            if self.history:
                r = self.history[-1]
                print('+' + '-' * len(line) + '+')
                line = '|'.join(str(r[x]).center(24) for x in idx)
                print('|' + line + '|')
            print('+' + '-' * len(line) + '+')

        return ret


class BayesianSearcher(Searcher):
    """ Class to search for neural architectures using Bayesian search strategy.

    Attribute:
        optimizer: An instance of BayesianOptimizer.
        t_min: A float. The minimum temperature during simulated annealing.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss,
                 generators, verbose, trainer_args=None,
                 default_model_len=None, default_model_width=None,
                 t_min=None, skip_conn=True):
        super(BayesianSearcher, self).__init__(n_output_node, input_shape,
                                               path, metric, loss,
                                               generators, verbose,
                                               trainer_args,
                                               default_model_len,
                                               default_model_width,
                                               skip_conn)
        if t_min is None:
            t_min = Constant.T_MIN
        self.optimizer = BayesianOptimizer(self, t_min, metric, skip_conn=self.skip_conn)

    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.

        Returns:
            list of 2-element tuples: generated_graph and other_info,
            for bayesian searcher the length of list is 1.
            generated_graph: An instance of Graph.
            other_info: Anything to be saved in the training queue together with the architecture.

        """
        remaining_time = self._timeout - time.time()
        generated_graph, new_father_id = self.optimizer.generate(self.descriptors,
                                                                 remaining_time, multiprocessing_queue)
        if new_father_id is None:
            new_father_id = 0
            generated_graph = self.generators[0](self.n_classes, self.input_shape). \
                generate(self.default_model_len, self.default_model_width)

        return [(generated_graph, new_father_id)]

    def update(self, other_info, model_id, graph, metric_value):
        """ Update the controller with evaluation result of a neural architecture.

        Args:
            other_info: Anything. In our case it is the father ID in the search tree.
            model_id: An integer.
            graph: An instance of Graph. The trained neural architecture.
            metric_value: The final evaluated metric value.
        """
        father_id = other_info
        self.optimizer.fit([graph.extract_descriptor()], [metric_value])
        self.optimizer.add_child(father_id, model_id)


def train(q, graph, train_data, test_data, trainer_args, metric, loss, verbose, path):
    """Train the neural architecture."""
    try:
        model = graph.produce_model()
        loss, metric_value = Backend.get_model_trainer(model=model,
                                                       path=path,
                                                       train_data=train_data,
                                                       test_data=test_data,
                                                       metric=metric,
                                                       loss_function=loss,
                                                       verbose=verbose).train_model(**trainer_args)
        model.set_weight_to_graph()
        if q:
            q.put((metric_value, loss, model.graph))
        return metric_value, loss, model.graph
    except RuntimeError as e:
        if not re.search('out of memory', str(e)):
            raise e
        if verbose:
            print('\nCurrent model size is too big. Discontinuing training this model to search for other models.')
        Constant.MAX_MODEL_SIZE = graph.size() - 1
        if q:
            q.put((None, None, None))
        return None, None, None
    except TimeoutError as exp:
        logging.warning("TimeoutError occurred at train() : {0}".format(str(exp)))
        if q:
            q.put((None, None, None))
        return None, None, None
    except Exception as exp:
        logging.warning("Exception occurred at train() : {0}".format(str(exp)))
        if verbose:
            print("Exception occurred at train() : {0}".format(str(exp)))
        if q:
            q.put((None, None, None))
        return None, None, None
