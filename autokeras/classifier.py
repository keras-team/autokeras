import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from autokeras import constant
from autokeras.preprocessor import OneHotEncoder
from autokeras.search import HillClimbingSearcher, RandomSearcher
from autokeras.utils import ensure_dir, reset_weights, ModelTrainer, has_file


def load_from_path(path=constant.DEFAULT_SAVE_PATH):
    """Load classifier that has been saved before

    The Classifier will be saved after fitting, so you can load it later instead of training again,
    which can save time.

    Args:
        path: the location in which the classifier has been saved

    Returns:
        The classifier that has been saved before
    """
    classifier = pickle.load(open(os.path.join(path, 'classifier'), 'rb'))
    classifier.path = path
    classifier.searcher = pickle.load(open(os.path.join(path, 'searcher'), 'rb'))
    return classifier


class ClassifierBase:
    """Base class of Classifier

    ClassifierBase is the base class of all classifier classes, classifier is used
    to train and predict data

    Attributes:
        y_encoder: encoder for y_train(array of category labels)
        verbose: verbosity mode
        searcher: a class that is used to find best model
        searcher_type: climb or random
        path: place that stores classifier
        model_id: identifier of one model
    """
    def __init__(self, verbose=False, searcher_type=None, path=constant.DEFAULT_SAVE_PATH):
        """Init ClassifierBase with verbose, searcher_type, path

        The classifier will be loaded from file if it has been saved before.
        """
        if has_file(os.path.join(path, 'classifier')):
            classifier = pickle.load(open(os.path.join(path, 'classifier'), 'rb'))
            classifier.searcher = pickle.load(open(os.path.join(path, 'searcher'), 'rb'))
            self.__dict__ = classifier.__dict__
        else:
            self.y_encoder = None
            self.verbose = verbose
            self.searcher = None
            self.searcher_type = searcher_type
            # self.history = []
            self.path = path
            self.model_id = None
            ensure_dir(path)

    def _validate(self, x_train, y_train):
        """Check x_train's type and the shape of x_train, y_train"""
        try:
            x_train = x_train.astype('float64')
        except ValueError:
            raise ValueError('x_train should only contain numerical data.')

        if len(x_train.shape) < 2:
            raise ValueError('x_train should at least has 2 dimensions.')

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError('x_train and y_train should have the same number of instances.')

    def fit(self, x_train, y_train):
        """Find the id of best model

        Format the input, and split the dataset into training and testing set,
        save the classifier and return id of the best model
        """
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
        """Return predict result for the testing data"""
        model = self.searcher.load_best_model()
        return self.y_encoder.inverse_transform(model.predict(x_test, verbose=self.verbose))

    def summary(self):
        """Print the summary of the best model"""
        model = self.searcher.load_best_model()
        model.summary()

    def _get_searcher_class(self):
        """return searcher class based on the searcher_type"""
        if self.searcher_type == 'climb':
            return HillClimbingSearcher
        elif self.searcher_type == 'random':
            return RandomSearcher
        return None

    def evaluate(self, x_test, y_test):
        """Return the error between predict value and test value"""
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    def cross_validate(self, x_all, y_all, n_splits):
        """Do the n_splits cross validation for the input"""
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
    """Normal classifier class inherited from ClassifierBase class"""
    def __init__(self):
        """Init Classifier"""
        super().__init__()

    def _validate(self, x_train, y_train):
        """check x_train's type and the shape of x_train, y_train"""
        super()._validate(x_train, y_train)


class ImageClassifier(ClassifierBase):
    """Image classifier class inherited from ClassifierBase class

    It's used to do image classification
    """
    def __init__(self, verbose=True, searcher_type='climb', path=constant.DEFAULT_SAVE_PATH):
        """Init ImageClassifier with verbose, searcher_type, path"""
        super().__init__(verbose, searcher_type, path)
