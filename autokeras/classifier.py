import multiprocessing
import os
import pickle
import csv
import time
import tensorflow as tf

import scipy.ndimage as ndimage

import numpy as np
from keras import backend
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from autokeras import constant
from autokeras.graph import Graph
from autokeras.preprocessor import OneHotEncoder
from autokeras.search import BayesianSearcher, train
from autokeras.utils import ensure_dir, ModelTrainer, has_file, pickle_from_file, pickle_to_file


def _validate(x_train, y_train):
    """Check x_train's type and the shape of x_train, y_train."""
    try:
        x_train = x_train.astype('float64')
    except ValueError:
        raise ValueError('x_train should only contain numerical data.')

    if len(x_train.shape) < 2:
        raise ValueError('x_train should at least has 2 dimensions.')

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError('x_train and y_train should have the same number of instances.')


def run_searcher_once(x_train, y_train, x_test, y_test, path):
    if constant.LIMIT_MEMORY:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        backend.set_session(sess)
    searcher = pickle_from_file(os.path.join(path, 'searcher'))
    searcher.search(x_train, y_train, x_test, y_test)


def read_csv_file(csv_file_path):
    """
    Read the cvs file and returns two seperate list containing images name and their labels
    :param csv_file_path: Path to the CVS file.
    :return: img_file_names list containing images names and img_label list containing their respective labels
    """
    img_file_names = []
    img_labels = []
    with open(csv_file_path, 'r') as images_path:
        path_list = csv.DictReader(images_path)
        fieldnames = path_list.fieldnames
        for path in path_list:
            img_file_names.append(path[fieldnames[0]])
            img_labels.append(path[fieldnames[1]])
    return img_file_names, img_labels


def read_images(img_file_names, images_dir_path):
    """
    Reads the images from the path and return there numpy.ndarray instance
    :param img_file_names: List containing images names
    :param images_dir_path: Path to directory containing images
    :return: Returns a numpy.ndarray instance containing the training data.
    """
    x_train = []
    if os.path.isdir(images_dir_path):
        for img_file in img_file_names:
            img_path = os.path.join(images_dir_path, img_file)
            if os.path.exists(img_path):
                img = ndimage.imread(fname=img_path)
                if len(img.shape) < 3:
                    img = img[..., np.newaxis]
                x_train.append(img)
            else:
                raise ValueError("%s image does not exist" % img_file)
    else:
        raise ValueError("Directory containing images does not exist")
    return np.asanyarray(x_train)


class ClassifierBase:
    """Base class of Classifier.

    ClassifierBase is the base class of all classifier classes, classifier is used
    to train and predict data.

    Attributes:
        y_encoder: An instance of OneHotEncoder for y_train (array of categorical labels).
        verbose: A boolean value indicating the verbosity mode.
        searcher: An instance of one of the subclasses of Searcher. It search different
            neural architecture to find the best model.
        searcher_type: The type of searcher to use. It must be 'climb' or 'random'.
        path: A path to the directory to save the classifier.
    """

    def __init__(self, verbose=False, searcher_type=None, path=constant.DEFAULT_SAVE_PATH, resume=False,
                 searcher_args=None):
        """Initialize the instance.

        The classifier will be loaded from file if the directory in 'path' has a saved classifier.
        Otherwise it would create a new one.
        """
        if searcher_args is None:
            searcher_args = {}

        if has_file(os.path.join(path, 'classifier')) and resume:
            classifier = pickle_from_file(os.path.join(path, 'classifier'))
            self.__dict__ = classifier.__dict__
            self.path = path
        else:
            self.y_encoder = None
            self.verbose = verbose
            self.searcher = False
            self.searcher_type = searcher_type
            self.path = path
            self.searcher_args = searcher_args
            ensure_dir(path)

    def fit(self, x_train=None, y_train=None, csv_file_path=None, images_path=None, time_limit=None):
        """Find the best model.

        Format the input, and split the dataset into training and testing set,
        save the classifier and find the best model.

        Args:
            time_limit:
            x_train: An numpy.ndarray instance contains the training data.
            y_train: An numpy.ndarray instance contains the label of the training data.
            csv_file_path: CVS file path
            images_path: Path where images exist
        """

        if y_train is None:
            y_train = []
        if x_train is None:
            x_train = []
        if csv_file_path is not None:
            img_file_name, y_train = read_csv_file(csv_file_path)
            if images_path is not None:
                x_train = read_images(img_file_name, images_path)
            else:
                raise ValueError('Directory containing images is not provided')

        x_train = np.array(x_train)
        y_train = np.array(y_train).flatten()

        _validate(x_train, y_train)

        # Transform y_train.
        if self.y_encoder is None:
            self.y_encoder = OneHotEncoder()
            self.y_encoder.fit(y_train)

        y_train = self.y_encoder.transform(y_train)

        # Create the searcher and save on disk
        if not self.searcher:
            input_shape = x_train.shape[1:]
            n_classes = self.y_encoder.n_classes
            self.searcher_args['n_classes'] = n_classes
            self.searcher_args['input_shape'] = input_shape
            self.searcher_args['path'] = self.path
            self.searcher_args['verbose'] = self.verbose
            searcher = self._get_searcher_class()(**self.searcher_args)
            self.save_searcher(searcher)
            self.searcher = True

        # Divide training data into training and testing data.
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

        pickle.dump(self, open(os.path.join(self.path, 'classifier'), 'wb'))
        pickle_to_file(self, os.path.join(self.path, 'classifier'))

        if time_limit is None:
            time_limit = 24*60*60

        start_time = time.time()
        while time.time() - start_time <= time_limit:
            run_searcher_once(x_train, y_train, x_test, y_test, self.path)
            if len(self.load_searcher().history) >= constant.MAX_MODEL_NUM:
                break

    def predict(self, x_test):
        """Return predict result for the testing data.

        Args:
            x_test: An instance of numpy.ndarray contains the testing data.
        """
        if constant.LIMIT_MEMORY:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            sess.run(init)
            backend.set_session(sess)
        model = self.load_searcher().load_best_model().produce_model()
        return self.y_encoder.inverse_transform(model.predict(x_test, ))

    def summary(self):
        """Print the summary of the best model."""
        model = self.load_searcher().load_best_model()
        model.summary()

    def _get_searcher_class(self):
        """Return searcher class based on the 'searcher_type'."""
        return BayesianSearcher

    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and test_y."""
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    def cross_validate(self, x_all, y_all, n_splits, trainer_args=None):
        """Do the n_splits cross-validation for the input."""
        if trainer_args is None:
            trainer_args = {}

        if constant.LIMIT_MEMORY:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            sess.run(init)
            backend.set_session(sess)

        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=7)
        ret = []
        y_raw_all = y_all
        y_all = self.y_encoder.transform(y_all)
        model = self.load_searcher().load_best_model().produce_model()
        for train, test in k_fold.split(x_all, y_raw_all):
            graph = Graph(model, False)
            backend.clear_session()
            model = graph.produce_model()
            ModelTrainer(model,
                         x_all[train],
                         y_all[train],
                         x_all[test],
                         y_all[test], False).train_model(**trainer_args)
            scores = model.evaluate(x_all[test], y_all[test], verbose=self.verbose)
            if self.verbose:
                print('Score:', scores[1])
            ret.append(scores[1] * 100)
        return np.array(ret)

    def save_searcher(self, searcher):
        pickle.dump(searcher, open(os.path.join(self.path, 'searcher'), 'wb'))

    def load_searcher(self):
        return pickle_from_file(os.path.join(self.path, 'searcher'))

    def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=False):
        if trainer_args is None:
            trainer_args = {}
        y_train = self.y_encoder.transform(y_train)
        y_test = self.y_encoder.transform(y_test)
        searcher = self.load_searcher()
        graph = searcher.load_best_model()
        if retrain:
            graph.weighted = False
        _, _1, graph = train((graph, x_train, y_train, x_test, y_test, trainer_args, None))
        searcher.replace_model(graph, searcher.get_best_model_id())


class ImageClassifier(ClassifierBase):
    """Image classifier class inherited from ClassifierBase class.

    It is used for image classification. It searches convolutional neural network architectures
    for the best configuration for the dataset.
    """

    def __init__(self, verbose=True, searcher_type='bayesian', path=constant.DEFAULT_SAVE_PATH, resume=False,
                 searcher_args=None):
        super().__init__(verbose,
                         searcher_type,
                         path,
                         resume=resume,
                         searcher_args=searcher_args)
