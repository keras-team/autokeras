import numpy as np
import pickle
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from autokeras import constant
from autokeras.search import HillClimbingSearcher, RandomSearcher
from autokeras.preprocessor import OneHotEncoder
from autokeras.utils import ensure_dir, reset_weights, ModelTrainer


def load_from_path(path=constant.DEFAULT_SAVE_PATH):
    classifier = pickle.load(open(os.path.join(path, 'classifier'), 'rb'))
    classifier.path = path
    classifier.searcher = pickle.load(open(os.path.join(path, 'searcher'), 'rb'))
    return classifier


class ClassifierBase:
    def __init__(self, verbose=False, searcher_type=None, path=constant.DEFAULT_SAVE_PATH):
        self.y_encoder = None
        self.verbose = verbose
        self.searcher = None
        self.searcher_type = searcher_type
        # self.history = []
        self.path = path
        self.model_id = None
        ensure_dir(path)

    def _validate(self, x_train, y_train):
        try:
            x_train = x_train.astype('float64')
        except ValueError:
            raise ValueError('x_train should only contain numerical data.')

        if len(x_train.shape) < 2:
            raise ValueError('x_train should at least has 2 dimensions.')

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError('x_train and y_train should have the same number of instances.')

    def fit(self, x_train, y_train):
        x_train = np.array(x_train)
        y_train = np.array(y_train).flatten()

        self._validate(x_train, y_train)

        # Transform y_train.
        if self.y_encoder is None:
            self.y_encoder = OneHotEncoder()
            self.y_encoder.fit(y_train)

        y_train = self.y_encoder.transform(y_train)

        if self.searcher is None:
            input_shape = x_train.shape[1:]
            n_classes = self.y_encoder.n_classes
            self.searcher = self._get_searcher_class()(n_classes, input_shape, self.path, self.verbose)

        # Divide training data into training and testing data.
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

        pickle.dump(self, open(os.path.join(self.path, 'classifier'), 'wb'))
        self.model_id = self.searcher.search(x_train, y_train, x_test, y_test)

    def predict(self, x_test):
        model = self.searcher.load_best_model()
        return self.y_encoder.inverse_transform(model.predict(x_test, verbose=self.verbose))

    def summary(self):
        model = self.searcher.load_best_model()
        model.summary()

    def _get_searcher_class(self):
        if self.searcher_type == 'climb':
            return HillClimbingSearcher
        elif self.searcher_type == 'random':
            return RandomSearcher
        return None

    def evaluate(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    def cross_validate(self, x_all, y_all, n_splits):
        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=7)
        scores = []
        y_raw_all = y_all
        y_all = self.y_encoder.transform(y_all)
        for train, test in k_fold.split(x_all, y_raw_all):
            model = self.searcher.load_best_model()
            reset_weights(model)
            ModelTrainer(model, x_all[train], y_all[train], x_all[test], y_all[test], self.verbose).train_model()
            scores = model.evaluate(x_all[test], y_all[test], verbose=self.verbose)
            scores.append(scores[1] * 100)
        return np.array(scores)


class Classifier(ClassifierBase):
    def __init__(self):
        super().__init__()

    def _validate(self, x_train, y_train):
        super()._validate(x_train, y_train)


class ImageClassifier(ClassifierBase):
    def __init__(self, verbose=True, searcher_type='climb', path=constant.DEFAULT_SAVE_PATH):
        super().__init__(verbose, searcher_type, path)
