import os
import pickle
import csv
import time
from functools import reduce

import torch

import scipy.ndimage as ndimage

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from autokeras.constant import Constant
from autokeras.preprocessor import OneHotEncoder, DataTransformer
from autokeras.search import BayesianSearcher, train
from autokeras.utils import ensure_dir, has_file, pickle_from_file, pickle_to_file

class Classifier:
    """The base classifier class.

    It is the base class for all the classifiers. It searches neural network architectures
    for the best configuration for the dataset.

    Attributes:
        path: A path to the directory to save the classifier.
        y_encoder: An instance of OneHotEncoder for `y_train` (array of categorical labels).
        verbose: A boolean value indicating the verbosity mode.
        searcher: An instance of BayesianSearcher. It searches different
            neural architecture to find the best model.
        searcher_args: A dictionary containing the parameters for the searcher's __init__ function.
        augment: A boolean value indicating whether the data needs augmentation.
    """
    def __init__(self, verbose=False, path=Constant.DEFAULT_SAVE_PATH, resume=False, \
    	searcher_args=None, augment=None):
    	"""Initialize the instance.

        The classifier will be loaded from the files in 'path' if parameter 'resume' is True.
        Otherwise it would create a new one.

        Args:
            verbose: A boolean of whether the search process will be printed to stdout.
            path: A string. The path to a directory, where the intermediate results are saved.
            resume: A boolean. If True, the classifier will continue to previous work saved in path.
                Otherwise, the classifier will start a new search.
            augment: A boolean value indicating whether the data needs augmentation.

        """
    	self.verbose = verbose
    	self.path = path
    	self.searcher_args = searcher_args
    	self.augment = augment

    def fit(self, x_train=None, y_train=None, time_limit=None):
    	"""Find the best neural architecture and train it.

        Based on the given dataset, the function will find the best neural architecture for it.
        The dataset is in numpy.ndarray format.
        So they training data should be passed through `x_train`, `y_train`.

        Args:
            x_train: A numpy.ndarray instance containing the training data.
            y_train: A numpy.ndarray instance containing the label of the training data.
            time_limit: The time limit for the search in seconds.
        """
    	pass
    def predict(self, x_test):
    	"""Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the results.
        """
    	pass
    def evaluate(self, x_test, y_test):
    	"""Return the accuracy score between predict value and `y_test`."""
    	pass