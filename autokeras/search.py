import os
import re
import time

import torch

from autokeras.constant import Constant
from autokeras.bayesian import edit_distance, BayesianOptimizer
from autokeras.generator import CnnGenerator
from autokeras.net_transformer import default_transform
from autokeras.utils import pickle_to_file, pickle_from_file, verbose_print
from autokeras.model_trainer import ModelTrainer

import torch.multiprocessing as mp


class Searcher:
    """Base class of all searcher classes.

    This class is the base class of all searcher classes,
    every searcher class can override its search function
    to implements its strategy.

    Attributes:
        n_classes: Number of classes in the target classification task.
        input_shape: Arbitrary, although all dimensions in the input shaped must be fixed.
            Use the keyword argument `input_shape` (tuple of integers, does not include the batch axis)
            when using this layer as the first layer in a model.
        verbose: Verbosity mode.
        history: A list that stores the performance of model. Each element in it is a dictionary of 'model_id',
            'loss', and 'metric_value'.
        path: A string. The path to the directory for saving the searcher.
        model_count: An integer. the total number of neural networks in the current searcher.
        descriptors: A dictionary of all the neural network architectures searched.
        trainer_args: A dictionary. The params for the constructor of ModelTrainer.
        default_model_len: An integer. Number of convolutional layers in the initial architecture.
        default_model_width: An integer. The number of filters in each layer in the initial architecture.
        search_tree: The data structure for storing all the searched architectures in tree structure.
        training_queue: A list of the generated architectures to be trained.
        x_queue: A list of trained architectures not updated to the gpr.
        y_queue: A list of trained architecture performances not updated to the gpr.
        beta: A float. The beta in the UCB acquisition function.
        t_min: A float. The minimum temperature during simulated annealing.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss, verbose,
                 trainer_args=None,
                 default_model_len=Constant.MODEL_LEN,
                 default_model_width=Constant.MODEL_WIDTH,
                 beta=Constant.BETA,
                 kernel_lambda=Constant.KERNEL_LAMBDA,
                 t_min=None):
        """Initialize the BayesianSearcher.

        Args:
            n_output_node: An integer, the number of classes.
            input_shape: A tuple. e.g. (28, 28, 1).
            path: A string. The path to the directory to save the searcher.
            verbose: A boolean. Whether to output the intermediate information to stdout.
            trainer_args: A dictionary. The params for the constructor of ModelTrainer.
            default_model_len: An integer. Number of convolutional layers in the initial architecture.
            default_model_width: An integer. The number of filters in each layer in the initial architecture.
            beta: A float. The beta in the UCB acquisition function.
            kernel_lambda: A float. The balance factor in the neural network kernel.
            t_min: A float. The minimum temperature during simulated annealing.
        """
        if trainer_args is None:
            trainer_args = {}
        self.n_classes = n_output_node
        self.input_shape = input_shape
        self.verbose = verbose
        self.history = []
        self.metric = metric
        self.loss = loss
        self.path = path
        self.model_count = 0
        self.descriptors = []
        self.trainer_args = trainer_args
        self.default_model_len = default_model_len
        self.default_model_width = default_model_width
        if 'max_iter_num' not in self.trainer_args:
            self.trainer_args['max_iter_num'] = Constant.SEARCH_MAX_ITER

        self.search_tree = SearchTree()
        self.training_queue = []
        self.x_queue = []
        self.y_queue = []
        if t_min is None:
            t_min = Constant.T_MIN
        self.bo = BayesianOptimizer(self, t_min, metric, kernel_lambda, beta)

    def load_model_by_id(self, model_id):
        return pickle_from_file(os.path.join(self.path, str(model_id) + '.h5'))

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
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.h5'))

    def add_model(self, metric_value, loss, graph, model_id):
        if self.verbose:
            print('\nSaving model.')

        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.h5'))

        # Update best_model text file
        ret = {'model_id': model_id, 'loss': loss, 'metric_value': metric_value}
        self.history.append(ret)
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
            for i, r in enumerate(self.history):
                print('+' + '-' * len(line) + '+')
                line = '|'.join(str(r[x]).center(24) for x in idx)
                print('|' + line + '|')
            print('+' + '-' * len(line) + '+')

        descriptor = graph.extract_descriptor()
        self.x_queue.append(descriptor)
        self.y_queue.append(metric_value)

        return ret

    def init_search(self):
        if self.verbose:
            print('\nInitializing search.')
        graph = CnnGenerator(self.n_classes,
                             self.input_shape).generate(self.default_model_len,
                                                        self.default_model_width)
        model_id = self.model_count
        self.model_count += 1
        self.training_queue.append((graph, -1, model_id))
        self.descriptors.append(graph.extract_descriptor())
        for child_graph in default_transform(graph):
            child_id = self.model_count
            self.model_count += 1
            self.training_queue.append((child_graph, model_id, child_id))
            self.descriptors.append(child_graph.extract_descriptor())
        if self.verbose:
            print('Initialization finished.')

    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        start_time = time.time()
        torch.cuda.empty_cache()
        if not self.history:
            self.init_search()

        # Start the new process for training.
        graph, father_id, model_id = self.training_queue.pop(0)
        if self.verbose:
            print('\n')
            print('+' + '-' * 46 + '+')
            print('|' + 'Training model {}'.format(model_id).center(46) + '|')
            print('+' + '-' * 46 + '+')
        mp.set_start_method('spawn', force=True)
        pool = mp.Pool(1)
        try:
            train_results = pool.map_async(train, [(graph, train_data, test_data, self.trainer_args,
                                                    os.path.join(self.path, str(model_id) + '.png'),
                                                    self.metric, self.loss, self.verbose)])

            # Do the search in current thread.
            searched = False
            new_graph = None
            new_father_id = None
            if not self.training_queue:
                searched = True

                while new_father_id is None:
                    remaining_time = timeout - (time.time() - start_time)
                    new_graph, new_father_id = self.bo.optimize_acq(self.search_tree.adj_list.keys(),
                                                                    self.descriptors,
                                                                    remaining_time)
                new_model_id = self.model_count
                self.model_count += 1
                self.training_queue.append((new_graph, new_father_id, new_model_id))
                self.descriptors.append(new_graph.extract_descriptor())

            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError
            metric_value, loss, graph = train_results.get(timeout=remaining_time)[0]

            if self.verbose and searched:
                verbose_print(new_father_id, new_graph)

            self.add_model(metric_value, loss, graph, model_id)
            self.search_tree.add_child(father_id, model_id)
            self.bo.fit(self.x_queue, self.y_queue)
            self.x_queue = []
            self.y_queue = []

            pickle_to_file(self, os.path.join(self.path, 'searcher'))
            self.export_json(os.path.join(self.path, 'history.json'))

        except (mp.TimeoutError, TimeoutError) as e:
            raise TimeoutError from e
        except RuntimeError as e:
            if not re.search('out of memory', str(e)):
                raise e
            if self.verbose:
                print('out of memory')
            Constant.MAX_MODEL_SIZE = graph.size() - 1
            return
        finally:
            # terminate and join the subprocess to prevent any resource leak
            pool.close()
            pool.join()

    def export_json(self, path):
        data = dict()

        networks = []
        for model_id in range(self.model_count - len(self.training_queue)):
            networks.append(self.load_model_by_id(model_id).extract_descriptor().to_json())

        tree = self.search_tree.get_dict()

        # Saving the data to file.
        data['networks'] = networks
        data['tree'] = tree
        import json
        with open(path, 'w') as fp:
            json.dump(data, fp)


class SearchTree:
    def __init__(self):
        self.root = None
        self.adj_list = {}

    def add_child(self, u, v):
        if u == -1:
            self.root = v
            self.adj_list[v] = []
            return
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
        if v not in self.adj_list:
            self.adj_list[v] = []

    def get_dict(self, u=None):
        if u is None:
            return self.get_dict(self.root)
        children = []
        for v in self.adj_list[u]:
            children.append(self.get_dict(v))
        ret = {'name': u, 'children': children}
        return ret


def train(args):
    graph, train_data, test_data, trainer_args, path, metric, loss, verbose = args
    model = graph.produce_model()
    # if path is not None:
    #     plot_model(model, to_file=path, show_shapes=True)
    loss, metric_value = ModelTrainer(model=model,
                                      train_data=train_data,
                                      test_data=test_data,
                                      metric=metric,
                                      loss_function=loss,
                                      verbose=verbose).train_model(**trainer_args)
    model.set_weight_to_graph()
    return metric_value, loss, model.graph


def same_graph(des1, des2):
    return edit_distance(des1, des2, 1) == 0
