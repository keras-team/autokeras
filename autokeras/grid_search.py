import os
import queue
import re
import time
import torch
import torch.multiprocessing as mp
import logging
from copy import deepcopy
from functools import total_ordering
from queue import PriorityQueue

import numpy as np
import math

from scipy.linalg import cholesky, cho_solve, solve_triangular, LinAlgError
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import rbf_kernel

from autokeras.constant import Constant
from autokeras.net_transformer import transform
from autokeras.nn.layers import is_layer, LayerType


from datetime import datetime
from autokeras.search import Searcher
from autokeras.bayesian import BayesianOptimizer
from autokeras.constant import Constant
from autokeras.nn.model_trainer import ModelTrainer
from autokeras.utils import pickle_to_file, pickle_from_file, verbose_print, get_system, assert_search_space


class Grid_Searcher(Searcher):
    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose, search_space,
                 trainer_args=None,
                 t_min=None):
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
            t_min: A float. The minimum temperature during simulated annealing.
        """
        if trainer_args is None:
            trainer_args = {}
        self.n_classes = n_output_node
        self.input_shape = input_shape
        self.verbose = verbose
        self.history = []
        self.path = path
        self.metric = metric
        self.loss = loss
        self.generators = generators
        self.model_count = 0
        self.descriptors = []
        self.trainer_args = trainer_args

        self.search_space, self.search_dimensions = assert_search_space(search_space)
        self.search_space_counter = 0
        #self.length_range = model_length_range #default_model_len if default_model_len is not None else Constant.MODEL_LEN
        #self.width_range = model_width_range #default_model_width if default_model_width is not None else Constant.MODEL_WIDTH
        if 'max_iter_num' not in self.trainer_args:
            self.trainer_args['max_iter_num'] = Constant.SEARCH_MAX_ITER
        #self.search_dimensions = 0
        #self.width_counter = 0
        self.training_queue = []
        self.x_queue = []
        self.y_queue = []
        # if t_min is None:
        #     t_min = Constant.T_MIN


        logging.basicConfig(filename=self.path+datetime.now().strftime('run_%d_%m_%Y : _%H_%M.log'),
                            format='%(asctime)s - %(filename)s - %(message)s', level=logging.DEBUG)

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

    def add_model(self, metric_value, loss, graph, model_id):
        """Append the information of evaluated architecture to history."""
        if self.verbose:
            print('\nSaving model.')

        graph.clear_operation_history()
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.graph'))

        ret = {'model_id': model_id, 'loss': loss, 'metric_value': metric_value}
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
    def init_search(self):
        """Call the generators to generate the initial architectures for the search."""
        if self.verbose:
            print('\nInitializing search.')
        for generator in self.generators:
            graph = generator(self.n_classes, self.input_shape). \
                generate(self.search_space[Constant.LENGTH_DIM][0], self.search_space[Constant.WIDTH_DIM][0])
            model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((graph, -1, model_id))
            self.descriptors.append(graph.extract_descriptor())

        if self.verbose:
            print('Initialization finished.')

    def search_space_exhausted(self):
        for i in range(len(self.search_space)):
            if self.search_dimensions[0][i] != self.search_dimensions[1][i]:
                return False
        return True

    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        """Run the search loop of training, generating and updating once.

        The function will run the training and generate in parallel.
        Then it will update the controller.
        The training is just pop out a graph from the training_queue and train it.
        The generate will call teh self.generate function.
        The update will call the self.update function.

        Args:
            train_data: An instance of DataLoader.
            test_data: An instance of Dataloader.
            timeout: An integer, time limit in seconds.
        """

        start_time = time.time()
        torch.cuda.empty_cache()
        if not self.history:
            self.init_search()

        if self.search_space_exhausted():
            return
        # Start the new process for training.
        graph, other_info, model_id = self.training_queue.pop(0)
        if self.verbose:
            print('\n')
            print('+' + '-' * 46 + '+')
            print('|' + 'Training model {}'.format(model_id).center(46) + '|')
            print('+' + '-' * 46 + '+')
        # Temporary solution to support GOOGLE Colab
        if get_system() == Constant.SYS_GOOGLE_COLAB:
            ctx = mp.get_context('fork')
        else:
            ctx = mp.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=train, args=(q, graph, train_data, test_data, self.trainer_args,
                                            self.metric, self.loss, self.verbose, self.path))
        try:
            p.start()
            # Do the search in current thread.
            searched = False
            generated_graph = None
            generated_other_info = None
            if not self.training_queue:
                searched = True
                #print("Update .. " + str(self.length_counter) + " ~ " + str(self.width_counter))
                remaining_time = timeout - (time.time() - start_time)
                generated_other_info, generated_graph = self.generate(remaining_time, q)
                new_model_id = self.model_count
                self.model_count += 1
                self.training_queue.append((generated_graph, generated_other_info, new_model_id))
                self.descriptors.append(generated_graph.extract_descriptor())

            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError
            metric_value, loss, graph = q.get(timeout=remaining_time)

            if self.verbose and searched:
                verbose_print(generated_other_info, generated_graph, new_model_id)

            if metric_value is not None:
                self.add_model(metric_value, loss, graph, model_id)


        except (TimeoutError, queue.Empty) as e:
            raise TimeoutError from e
        finally:
            # terminate and join the subprocess to prevent any resource leak
            p.terminate()
            p.join()


    def generate(self, remaining_time, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            remaining_time: The remaining time in seconds.
            multiprocessing_queue: the Queue for multiprocessing return value.

        Returns:
            other_info: Anything to be saved in the training queue together with the architecture.
            generated_graph: An instance of Graph.

        """
        grid = self.get_grid()
        generated_graph = self.generators[0](self.n_classes, self.input_shape). \
            generate(grid[Constant.LENGTH_DIM], grid[Constant.WIDTH_DIM])
        return 0, generated_graph

    def get_grid(self):
        self.search_space_counter += 1
        if self.search_space_counter < len(self.search_dimensions):
            return self.search_dimensions[self.search_space_counter]
        return None

def train(q, graph, train_data, test_data, trainer_args, metric, loss, verbose, path):
    """Train the neural architecture."""
    try:
        model = graph.produce_model()
        loss, metric_value = ModelTrainer(model=model,
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

@total_ordering
class Elem:
    """Elements to be sorted according to metric value."""

    def __init__(self, metric_value, father_id, graph):
        self.father_id = father_id
        self.graph = graph
        self.metric_value = metric_value

    def __eq__(self, other):
        return self.metric_value == other.metric_value

    def __lt__(self, other):
        return self.metric_value < other.metric_value


class ReverseElem(Elem):
    """Elements to be reversely sorted according to metric value."""

    def __lt__(self, other):
        return self.metric_value > other.metric_value