import os
import pickle
import numpy as np

from keras.losses import categorical_crossentropy
from keras.models import load_model
from keras.optimizers import Adadelta
from keras import backend

from autokeras import constant
from autokeras.bayesian import IncrementalGaussianProcess
from autokeras.generator import RandomConvClassifierGenerator, DefaultClassifierGenerator
from autokeras.graph import Graph, NetworkMorphismGraph
from autokeras.net_transformer import transform
from autokeras.stub import to_stub_model
from autokeras.utils import ModelTrainer
from autokeras.utils import extract_config
from autokeras.utils import has_file


class Searcher:
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
        history_configs: a list that stores all historical configuration
        history: a list that stores the performance of model
        path: place that store searcher
        model_count: the id of model
    """
    def __init__(self, n_classes, input_shape, path, verbose):
        """Init Searcher class with n_classes, input_shape, path, verbose

        The Searcher will be loaded from file if it has been saved before.
        """
        if has_file(os.path.join(path, "searcher")):
            searcher = pickle.load(open(os.path.join(path, 'searcher'), 'rb'))
            self.__dict__ = searcher.__dict__
        else:
            self.n_classes = n_classes
            self.input_shape = input_shape
            self.verbose = verbose
            self.history_configs = []
            self.history = []
            self.path = path
            self.model_count = 0

    def search(self, x_train, y_train, x_test, y_test):
        """an search strategy that will be overridden by children classes"""
        pass

    def load_model_by_id(self, model_id):
        return load_model(os.path.join(self.path, str(model_id) + '.h5'))

    def load_best_model(self):
        """return model with best accuracy"""
        return self.load_model_by_id(max(self.history, key=lambda x: x['accuracy'])['model_id'])

    def add_model(self, model, x_train, y_train, x_test, y_test):
        """add one model while will be trained to history list

        Returns:
            History object.
        """
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adadelta(),
                      metrics=['accuracy'])
        if self.verbose:
            model.summary()
        ModelTrainer(model, x_train, y_train, x_test, y_test, self.verbose).train_model()
        loss, accuracy = model.evaluate(x_test, y_test, verbose=self.verbose)
        model.save(os.path.join(self.path, str(self.model_count) + '.h5'))

        ret = {'model_id': self.model_count, 'loss': loss, 'accuracy': accuracy}
        self.history.append(ret)
        self.history_configs.append(extract_config(model))
        self.model_count += 1

        return ret


class RandomSearcher(Searcher):
    """Random Searcher class inherited from ClassifierBase class

    RandomSearcher implements its search function with random strategy
    """
    def __init__(self, n_classes, input_shape, path, verbose):
        """Init RandomSearcher with n_classes, input_shape, path, verbose"""
        super().__init__(n_classes, input_shape, path, verbose)

    def search(self, x_train, y_train, x_test, y_test):
        """Override parent's search function. First model is randomly generated"""
        while self.model_count < constant.MAX_MODEL_NUM:
            model = RandomConvClassifierGenerator(self.n_classes, self.input_shape).generate()
            self.add_model(model, x_train, y_train, x_test, y_test)
            pickle.dump(self, open(os.path.join(self.path, 'searcher'), 'wb'))
            backend.clear_session()

        return self.load_best_model()


class HillClimbingSearcher(Searcher):
    """HillClimbing Searcher class inherited from ClassifierBase class

    HillClimbing Searcher implements its search function with hill climbing strategy
    """
    def __init__(self, n_classes, input_shape, path, verbose):
        """Init HillClimbing Searcher with n_classes, input_shape, path, verbose"""
        super().__init__(n_classes, input_shape, path, verbose)

    def _remove_duplicate(self, models):
        """Remove the duplicate in the history_models"""
        ans = []
        for model_a in models:
            model_a_config = extract_config(model_a)
            if model_a_config not in self.history_configs:
                ans.append(model_a)
        return ans

    def search(self, x_train, y_train, x_test, y_test):
        """Override parent's search function. First model is randomly generated"""
        if not self.history:
            model = DefaultClassifierGenerator(self.n_classes, self.input_shape).generate()
            self.add_model(model, x_train, y_train, x_test, y_test)
            pickle.dump(self, open(os.path.join(self.path, 'searcher'), 'wb'))

        optimal_accuracy = 0.0
        while self.model_count < constant.MAX_MODEL_NUM:
            model = self.load_best_model()
            new_graphs = transform(Graph(to_stub_model(model)))
            new_models = []
            for graph in new_graphs:
                nm_graph = NetworkMorphismGraph(model)
                for args in graph.operation_history:
                    getattr(nm_graph, args[0])(*list(args[1:]))
                    new_models.append(nm_graph.produce_model())
            new_models = self._remove_duplicate(list(new_models))

            for model in new_models:
                if self.model_count < constant.MAX_MODEL_NUM:
                    self.add_model(model, x_train, y_train, x_test, y_test)
                    pickle.dump(self, open(os.path.join(self.path, 'searcher'), 'wb'))

            backend.clear_session()
            max_accuracy = max(self.history, key=lambda x: x['accuracy'])['accuracy']
            if max_accuracy <= optimal_accuracy:
                break
            optimal_accuracy = max_accuracy

        return self.load_best_model()


class BayesianSearcher(Searcher):

    def __init__(self, n_classes, input_shape, path, verbose):
        super().__init__(n_classes, input_shape, path, verbose)
        self.gpr = IncrementalGaussianProcess()
        self.search_tree = SearchTree()

    def search(self, x_train, y_train, x_test, y_test):
        if not self.history:
            model = DefaultClassifierGenerator(self.n_classes, self.input_shape).generate()
            history_item = self.add_model(model, x_train, y_train, x_test, y_test)
            self.search_tree.add_child(-1, history_item['model_id'])
            self.gpr.first_fit(Graph(model).extract_descriptor(), history_item['accuracy'])
            pickle.dump(self, open(os.path.join(self.path, 'searcher'), 'wb'))
            del model
            backend.clear_session()

        else:
            model_ids = self.search_tree.get_leaves()
            new_model, father_id = self.maximize_acq(model_ids)

            history_item = self.add_model(new_model, x_train, y_train, x_test, y_test)
            self.search_tree.add_child(father_id, history_item['model_id'])
            self.gpr.incremental_fit(Graph(new_model).extract_descriptor(), history_item['accuracy'])
            pickle.dump(self, open(os.path.join(self.path, 'searcher'), 'wb'))
            del new_model
            backend.clear_session()

        return self.load_best_model()

    def maximize_acq(self, model_ids):
        overall_max_acq_value = -1
        father_id = None
        target_graph = None

        # exploration
        for model_id in model_ids:
            model = self.load_model_by_id(model_id)
            graph = Graph(to_stub_model(model))
            graph.clear_operation_history()
            graphs = transform(graph)
            for temp_graph in graphs:
                temp_acq_value = self._acq(temp_graph)
                if temp_acq_value > overall_max_acq_value:
                    overall_max_acq_value = temp_acq_value
                    father_id = model_id
                    target_graph = temp_graph

        # exploitation
        for i in range(constant.ACQ_EXPLOITATION_DEPTH):
            graphs = transform(target_graph)
            for temp_graph in graphs:
                temp_acq_value = self._acq(temp_graph)
                if temp_acq_value > overall_max_acq_value:
                    overall_max_acq_value = temp_acq_value
                    target_graph = temp_graph

        model = self.load_model_by_id(father_id)
        nm_graph = NetworkMorphismGraph(model)
        for args in target_graph.operation_history:
            getattr(nm_graph, args[0])(*list(args[1:]))
        return nm_graph.produce_model(), father_id

    def _acq(self, graph):
        return self.gpr.predict(np.array([graph.extract_descriptor()]), )[0]


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
