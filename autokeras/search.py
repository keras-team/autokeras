import multiprocessing
import os
import random
from functools import total_ordering
from queue import PriorityQueue

import numpy as np
import math

from autokeras import constant
from autokeras.bayesian import IncrementalGaussianProcess
from autokeras.generator import DefaultClassifierGenerator
from autokeras.graph import Graph
from autokeras.net_transformer import transform
from autokeras.utils import ModelTrainer, pickle_to_file, pickle_from_file


class BayesianSearcher:
    """Base class of all searcher class

    This class is the base class of all searcher class,
    every searcher class can override its search function
    to implements its strategy

    Attributes:
        n_classes: number of classification
        input_shape: Arbitrary, although all dimensions in the input shaped must be fixed.
                   Use the keyword argument input_shape (tuple of integers, does not include the batch axis)
                   when using this layer as the first layer in a model.
        verbose: verbosity mode
        history: a list that stores the performance of model
        path: place that store searcher
        model_count: the id of model
    """

    def __init__(self, n_classes, input_shape, path, verbose,
                 trainer_args=None,
                 default_model_len=constant.MODEL_LEN,
                 default_model_width=constant.MODEL_WIDTH,
                 beta=constant.BETA,
                 kernel_lambda=constant.KERNEL_LAMBDA,
                 t_min=constant.T_MIN):
        """Init Searcher class with n_classes, input_shape, path, verbose

        The Searcher will be loaded from file if it has been saved before.
        """
        if trainer_args is None:
            trainer_args = {}
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.verbose = verbose
        self.history = []
        self.path = path
        self.model_count = 0
        self.descriptors = {}
        self.trainer_args = trainer_args
        self.default_model_len = default_model_len
        self.default_model_width = default_model_width
        if 'max_iter_num' not in self.trainer_args:
            self.trainer_args['max_iter_num'] = constant.SEARCH_MAX_ITER

        self.gpr = IncrementalGaussianProcess(kernel_lambda)
        self.search_tree = SearchTree()
        self.training_queue = []
        self.x_queue = []
        self.y_queue = []
        self.beta = beta
        self.t_min = t_min

    def load_model_by_id(self, model_id):
        return pickle_from_file(os.path.join(self.path, str(model_id) + '.h5'))

    def load_best_model(self):
        """return model with best accuracy"""
        return self.load_model_by_id(self.get_best_model_id())

    def get_accuracy_by_id(self, model_id):
        for item in self.history:
            if item['model_id'] == model_id:
                return item['accuracy']
        return None

    def get_best_model_id(self):
        return max(self.history, key=lambda x: x['accuracy'])['model_id']

    def replace_model(self, graph, model_id):
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.h5'))

    def add_model(self, accuracy, loss, graph, model_id):
        """add one model while will be trained to history list

        Returns:
            History object.
        """
        if self.verbose:
            print('Saving model.')

        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.h5'))
        # plot_model(model, to_file=os.path.join(self.path, str(model_id) + '.png'), show_shapes=True)

        # Update best_model text file

        if self.verbose:
            print('Model ID:', model_id)
            print('Loss:', loss)
            print('Accuracy', accuracy)

        ret = {'model_id': model_id, 'loss': loss, 'accuracy': accuracy}
        descriptor = graph.extract_descriptor()
        self.history.append(ret)
        if model_id == self.get_best_model_id():
            file = open(os.path.join(self.path, 'best_model.txt'), 'w')
            file.write('best model: ' + str(model_id))
            file.close()

        self.descriptors[descriptor] = True
        self.search_tree.add_child(-1, model_id)
        self.x_queue.append(descriptor)
        self.y_queue.append(accuracy)

        return ret

    def init_search(self):
        if self.verbose:
            print('Initializing search.')
        graph = DefaultClassifierGenerator(self.n_classes,
                                           self.input_shape).generate(self.default_model_len,
                                                                      self.default_model_width)
        model_id = self.model_count
        self.model_count += 1
        self.training_queue.append((graph, -1, model_id))
        for child_graph in transform(graph):
            child_id = self.model_count
            self.model_count += 1
            self.training_queue.append((child_graph, model_id, child_id))
        if self.verbose:
            print('Initialization finished.')

    def search(self, x_train, y_train, x_test, y_test):
        if not self.history:
            self.init_search()

        # Start the new process for training.
        graph, father_id, model_id = self.training_queue.pop(0)
        pool = multiprocessing.Pool(1)
        train_results = pool.map_async(train, [(graph, x_test, x_train, y_test, y_train, self.trainer_args)])

        # Do the search in current thread.
        if not self.training_queue:
            graph, father_id = self.maximize_acq()
            new_model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((graph, father_id, new_model_id))

        accuracy, loss, graph = train_results.get()[0]
        self.add_model(accuracy, loss, graph, model_id)
        self.gpr.fit(self.x_queue, self.y_queue)
        self.x_queue = []
        self.y_queue = []

        pickle_to_file(self, os.path.join(self.path, 'searcher'))

    def maximize_acq(self):
        model_ids = self.search_tree.adj_list.keys()
        target_graph = None
        father_id = None
        descriptors = self.descriptors

        pq = PriorityQueue()
        temp_list = []
        for model_id in model_ids:
            accuracy = self.get_accuracy_by_id(model_id)
            temp_list.append((accuracy, model_id))
        temp_list = sorted(temp_list)
        if len(temp_list) > 5:
            temp_list = temp_list[:-5]
        for accuracy, model_id in temp_list:
            graph = self.load_model_by_id(model_id)
            pq.put(Elem(accuracy, model_id, graph))

        t = 1.0
        t_min = self.t_min
        alpha = 0.9
        max_acq = -1
        while not pq.empty() and t > t_min:
            elem = pq.get()
            ap = math.exp((elem.accuracy - max_acq) / t)
            if ap > random.uniform(0, 1):
                graphs = transform(elem.graph)
                graphs = list(filter(lambda x: x.extract_descriptor() not in descriptors, graphs))
                if not graphs:
                    continue
                for temp_graph in graphs:
                    temp_acq_value = self.acq(temp_graph)
                    pq.put(Elem(temp_acq_value, elem.father_id, temp_graph))
                    descriptors[temp_graph.extract_descriptor()] = True
                    if temp_acq_value > max_acq:
                        max_acq = temp_acq_value
                        father_id = elem.father_id
                        target_graph = temp_graph
            t *= alpha

        nm_graph = self.load_model_by_id(father_id)
        if self.verbose:
            print('Father ID: ', father_id)
            print(target_graph.operation_history)
        for args in target_graph.operation_history:
            getattr(nm_graph, args[0])(*list(args[1:]))
        return nm_graph, father_id

    def acq(self, graph):
        mean, std = self.gpr.predict(np.array([graph.extract_descriptor()]))
        return mean + self.beta * std


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

    def get_leaves(self):
        ret = []
        for key, value in self.adj_list.items():
            if not value:
                ret.append(key)
        return ret


@total_ordering
class Elem:
    def __init__(self, accuracy, father_id, graph):
        self.father_id = father_id
        self.graph = graph
        self.accuracy = accuracy

    def __eq__(self, other):
        return self.accuracy == other.accuracy

    def __lt__(self, other):
        return self.accuracy < other.accuracy


def train(args):
    graph, x_test, x_train, y_test, y_train, trainer_args = args
    model = graph.produce_model()
    loss, accuracy = ModelTrainer(model,
                                  x_train,
                                  y_train,
                                  x_test,
                                  y_test,
                                  False).train_model(**trainer_args)
    return accuracy, loss, Graph(model, True)
