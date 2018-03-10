import os
import pickle

from keras.losses import categorical_crossentropy
from keras.models import load_model
from keras.optimizers import Adadelta

from autokeras import constant
from autokeras.generator import RandomConvClassifierGenerator, DefaultClassifierGenerator
from autokeras.net_transformer import transform
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
            model ID.
        """
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adadelta(),
                      metrics=['accuracy'])
        if self.verbose:
            model.summary()
        ModelTrainer(model, x_train, y_train, x_test, y_test, self.verbose).train_model()
        loss, accuracy = model.evaluate(x_test, y_test, verbose=self.verbose)
        model.save(os.path.join(self.path, str(self.model_count) + '.h5'))
        self.history.append({'model_id': self.model_count, 'loss': loss, 'accuracy': accuracy})
        self.history_configs.append(extract_config(model))
        self.model_count += 1
        pickle.dump(self, open(os.path.join(self.path, 'searcher'), 'wb'))
        return self.model_count - 1


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

        optimal_accuracy = 0.0
        while self.model_count < constant.MAX_MODEL_NUM:
            model = self.load_best_model()
            new_models = self._remove_duplicate(transform(model))

            for model in new_models:
                if self.model_count < constant.MAX_MODEL_NUM:
                    self.add_model(model, x_train, y_train, x_test, y_test)

            max_accuracy = max(self.history, key=lambda x: x['accuracy'])['accuracy']
            if max_accuracy <= optimal_accuracy:
                break
            optimal_accuracy = max_accuracy

        return self.load_best_model()


class BayesianSearcher(HillClimbingSearcher):

    def __init__(self, n_classes, input_shape, path, verbose):
        super().__init__(n_classes, input_shape, path, verbose)
        self.search_tree = SearchTree()

    def search(self, x_train, y_train, x_test, y_test):
        if not self.history:
            model = DefaultClassifierGenerator(self.n_classes, self.input_shape).generate()
            model_id = self.add_model(model, x_train, y_train, x_test, y_test)
            self.search_tree.add_child(-1, model_id)

        optimal_accuracy = 0.0
        while self.model_count < constant.MAX_MODEL_NUM:
            model_ids = self.search_tree.get_leaves()
            new_model, father_id = self.maximize_acq(model_ids)

            if self.model_count < constant.MAX_MODEL_NUM:
                new_model_id = self.add_model(new_model, x_train, y_train, x_test, y_test)
                self.search_tree.add_child(father_id, new_model_id)

            max_accuracy = max(self.history, key=lambda x: x['accuracy'])['accuracy']
            if max_accuracy <= optimal_accuracy:
                break
            optimal_accuracy = max_accuracy

        return self.load_best_model()

    def maximize_acq(self, model_ids):
        # TODO: implement it
        print(model_ids)
        # exploration
        for model_id in model_ids:
            self.load_model_by_id(model_id)
        # exploitation
        return self.load_best_model()


class SearchTree:
    # TODO: implement search tree
    def __init__(self):
        self.nodes = None

    def add_child(self, u, v):
        pass

    def get_leaves(self):
        return self.nodes
