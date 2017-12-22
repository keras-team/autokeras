import os
import pickle

from keras.losses import categorical_crossentropy
from keras.models import load_model
from keras.optimizers import Adadelta

from autokeras import constant
from autokeras.generator import RandomConvClassifierGenerator, DefaultClassifierGenerator
from autokeras.utils import extract_config
from autokeras.net_transformer import transform
from autokeras.utils import ModelTrainer


class Searcher:
    def __init__(self, n_classes, input_shape, path, verbose):
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.verbose = verbose
        self.history_configs = []
        self.history = []
        self.path = path
        self.model_count = 0

    def search(self, x_train, y_train, x_test, y_test):
        pass

    def load_best_model(self):
        pass

    def add_model(self, model, x_train, y_train, x_test, y_test):
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adadelta(),
                      metrics=['accuracy'])
        model.summary()
        ModelTrainer(model, x_train, y_train, x_test, y_test, self.verbose).train_model()
        loss, accuracy = model.evaluate(x_test, y_test)
        model.save(os.path.join(self.path, str(self.model_count) + '.h5'))
        self.history.append({'model_id': self.model_count, 'loss': loss, 'accuracy': accuracy})
        self.history_configs.append(extract_config(model))
        self.model_count += 1
        pickle.dump(self, open(os.path.join(self.path, 'searcher'), 'wb'))


class RandomSearcher(Searcher):
    def __init__(self, n_classes, input_shape, path, verbose):
        super().__init__(n_classes, input_shape, path, verbose)

    def search(self, x_train, y_train, x_test, y_test):
        # First model is randomly generated.
        while self.model_count < constant.MAX_MODEL_NUM:
            model = RandomConvClassifierGenerator(self.n_classes, self.input_shape).generate()
            self.add_model(model, x_train, y_train, x_test, y_test)

        return self.load_best_model()


class HillClimbingSearcher(Searcher):
    def __init__(self, n_classes, input_shape, path, verbose):
        super().__init__(n_classes, input_shape, path, verbose)

    def _remove_duplicate(self, models):
        """
        Remove the duplicate in the history_models
        :param models:
        :return:
        """
        ans = []
        for model_a in models:
            model_a_config = extract_config(model_a)
            if model_a_config not in self.history_configs:
                ans.append(model_a)
        return ans

    def search(self, x_train, y_train, x_test, y_test):
        # First model is randomly generated.
        if not self.history:
            model = DefaultClassifierGenerator(self.n_classes, self.input_shape).generate()
            self.add_model(model, x_train, y_train, x_test, y_test)

        optimal_accuracy = 0.0
        while self.model_count < constant.MAX_MODEL_NUM:
            model = self.load_best_model()
            new_models = self._remove_duplicate(transform(model))
            for model in new_models:
                self.history_configs.append(extract_config(model))

            for model in new_models:
                if self.model_count < constant.MAX_MODEL_NUM:
                    self.add_model(model, x_train, y_train, x_test, y_test)

            max_accuracy = max(self.history, key=lambda x: x['accuracy'])['accuracy']
            if max_accuracy <= optimal_accuracy:
                break
            optimal_accuracy = max_accuracy

        return self.load_best_model()

    def load_best_model(self):
        model_id = max(self.history, key=lambda x: x['accuracy'])['model_id']
        return load_model(os.path.join(self.path, str(model_id) + '.h5'))
