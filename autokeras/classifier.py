import os
import pickle
import csv
import time
import tensorflow as tf

import scipy.ndimage as ndimage

import numpy as np
from keras import backend
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from autokeras import constant
from autokeras.preprocessor import OneHotEncoder
from autokeras.search import BayesianSearcher, train
from autokeras.utils import ensure_dir, has_file, pickle_from_file, pickle_to_file


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
    """Read the cvs file and returns two seperate list containing images name and their labels

    Args:
        csv_file_path: Path to the CVS file.

    Returns:
        img_file_names list containing images names and img_label list containing their respective labels.
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
    """Reads the images from the path and return there numpy.ndarray instance

    Args:
        img_file_names: List containing images names
        images_dir_path: Path to directory containing images

    Returns:
        Returns a numpy.ndarray instance containing the training data.
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


def load_image_dataset(csv_file_path, images_path):
    """Load images from the files and labels from a csv file.

    Second, the dataset is a set of images and the labels are in a CSV file.
    The CSV file should contain two columns whose names are 'File Name' and 'Label'.
    The file names in the first column should match the file names of the images with extensions,
    e.g., .jpg, .png.
    The path to the CSV file should be passed through the csv_file_path.
    The path to the directory containing all the images should be passed through image_path.

    Args:
        csv_file_path: CVS file path.
        images_path: Path where images exist.

    Returns:
        x: Four dimensional numpy.ndarray. The channel dimension is the last dimension.
        y: The labels.
    """
    img_file_name, y = read_csv_file(csv_file_path)
    x = read_images(img_file_name, images_path)
    return np.array(x), np.array(y)


class ImageClassifier:
    """The image classifier class.

    It is used for image classification. It searches convolutional neural network architectures
    for the best configuration for the dataset.

    Attributes:
        path: A path to the directory to save the classifier.
        y_encoder: An instance of OneHotEncoder for y_train (array of categorical labels).
        verbose: A boolean value indicating the verbosity mode.
        searcher: An instance of BayesianSearcher. It search different
            neural architecture to find the best model.
        searcher_args: A dictionary containing the parameters for the searcher's __init__ function.
    """

    def __init__(self, verbose=False, path=constant.DEFAULT_SAVE_PATH, resume=False,
                 searcher_args=None):
        """Initialize the instance.

        The classifier will be loaded from the files in 'path' if parameter 'resume' is True.
        Otherwise it would create a new one.

        Args:
            verbose: An boolean of whether the search process will be printed to stdout.
            path: A string. The path to a directory, where the intermediate results are saved.
            resume: An boolean. If True, the classifier will continue to previous work saved in path.
                Otherwise, the classifier will start a new search.

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
            self.path = path
            self.searcher_args = searcher_args
            ensure_dir(path)

    def fit(self, x_train=None, y_train=None, time_limit=None):
        """Find the best neural architecture and train it.

        Based on the given dataset, the function will find the best neural architecture for it.
        The dataset is in numpy.ndarray format.
        So they training data should be passed through x_train, y_train.

        Args:
            x_train: An numpy.ndarray instance contains the training data.
            y_train: An numpy.ndarray instance contains the label of the training data.
            time_limit: The time limit for the search in seconds.
        """

        if y_train is None:
            y_train = []
        if x_train is None:
            x_train = []

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
            searcher = BayesianSearcher(**self.searcher_args)
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

        Returns:
            An numpy.ndarray containing the results.
        """
        if constant.LIMIT_MEMORY:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            sess.run(init)
            backend.set_session(sess)
        x_test = x_test.astype('float32') / 255
        model = self.load_searcher().load_best_model().produce_model()
        return self.y_encoder.inverse_transform(model.predict(x_test, ))

    def summary(self):
        """Print the summary of the best model."""
        model = self.load_searcher().load_best_model()
        model.summary()

    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and test_y."""
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    def save_searcher(self, searcher):
        pickle.dump(searcher, open(os.path.join(self.path, 'searcher'), 'wb'))

    def load_searcher(self):
        return pickle_from_file(os.path.join(self.path, 'searcher'))

    def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=False):
        """Final training after found the best architecture.

        Args:
            x_train: An numpy.ndarray of training data.
            y_train: An numpy.ndarray of training targets.
            x_test: An numpy.ndarray of testing data.
            y_test: An numpy.ndarray of testing targets.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructure.
            retrain: A boolean of whether reinitialize the weights of the model.
        """
        if trainer_args is None:
            trainer_args = {}
        y_train = self.y_encoder.transform(y_train)
        y_test = self.y_encoder.transform(y_test)
        searcher = self.load_searcher()
        graph = searcher.load_best_model()
        if retrain:
            graph.weighted = False
        _, _1, graph = train((graph, x_train, y_train, x_test, y_test, trainer_args, None))

    def export_keras_model(self, path, model_id=None):
        """Export the searched model as a Keras saved model.

        Args:
            path: A string. The path to the file to save.
            model_id: A integer. If not provided, the function will export the best model.
        """
        if model_id is None:
            model_id = self.get_best_model_id()
        graph = self.load_searcher().load_model_by_id(model_id)
        graph.produce_model().save(path)

    def get_best_model_id(self):
        """
        Returns:
            An integer. The best model id.
        """
        return self.load_searcher().get_best_model_id()
