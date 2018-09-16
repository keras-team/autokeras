import csv
import os
import pickle
import time
from abc import abstractmethod
from functools import reduce

import numpy as np
from scipy import ndimage
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from autokeras.loss_function import classification_loss, regression_loss
from autokeras.supervised import Supervised
from autokeras.constant import Constant
from autokeras.metric import Accuracy, MSE
from autokeras.preprocessor import OneHotEncoder, DataTransformer
from autokeras.search import Searcher, train
from autokeras.utils import ensure_dir, has_file, pickle_from_file, pickle_to_file, temp_folder_generator
from torch.utils.data import Dataset, DataLoader

from autokeras.image_supervised import ImageSupervised


def _validate(x_train, y_train):
    """Check `x_train`'s type and the shape of `x_train`, `y_train`."""
    try:
        x_train = x_train.astype('float64')
    except ValueError:
        raise ValueError('x_train should only contain numerical data.')

    if len(x_train.shape) < 2:
        raise ValueError('x_train should at least has 2 dimensions.')

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError('x_train and y_train should have the same number of instances.')


def run_searcher_once(train_data, test_data, path, timeout):
    if Constant.LIMIT_MEMORY:
        pass
    searcher = pickle_from_file(os.path.join(path, 'searcher'))
    searcher.search(train_data, test_data, timeout)


class CustomerDataset(Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.dataset)


class TextClassifier(ImageSupervised):
    @property
    def loss(self):
        return classification_loss

    def fit(self, x_train=None, y_train=None, batch_size=None , time_limit=None):
        """Find the best neural architecture and train it.

        Based on the given dataset, the function will find the best neural architecture for it.
        The dataset is in numpy.ndarray format.
        So they training data should be passed through `x_train`, `y_train`.

        Args:
            x_train: A numpy.ndarray instance containing the training data.
            y_train: A numpy.ndarray instance containing the label of the training data.
            time_limit: The time limit for the search in seconds.
        """
        if y_train is None:
            y_train = []
        if x_train is None:
            x_train = []

        x_train = np.array(x_train)

        _validate(x_train, y_train)

        # Transform x_train
        if self.data_transformer is None:
            self.data_transformer = DataTransformer(x_train, augment=self.augment)

        # Create the searcher and save on disk
        if not self.searcher:
            input_shape = x_train.shape[1:]
            self.searcher_args['n_output_node'] = y_train.shape[1]
            self.searcher_args['input_shape'] = input_shape
            self.searcher_args['path'] = self.path
            self.searcher_args['metric'] = self.metric
            self.searcher_args['loss'] = self.loss
            self.searcher_args['verbose'] = self.verbose
            searcher = Searcher(**self.searcher_args)
            self.save_searcher(searcher)
            self.searcher = True

        # Divide training data into training and testing data.
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                            test_size=min(Constant.VALIDATION_SET_SIZE,
                                                                          int(len(y_train) * 0.2)),
                                                            random_state=42)

        batch_size = 20
        # Wrap the data into DataLoaders
        x_train_transpose = torch.Tensor(x_train.transpose(0, 3, 1, 2))
        x_test_transpose = torch.Tensor(x_test.transpose(0, 3, 1, 2))
        train_data = DataLoader(CustomerDataset(x_train_transpose, y_train), batch_size=batch_size, shuffle=True)
        test_data = DataLoader(CustomerDataset(x_test_transpose, y_test), batch_size=batch_size, shuffle=True)

        # Save the classifier
        pickle.dump(self, open(os.path.join(self.path, 'classifier'), 'wb'))
        pickle_to_file(self, os.path.join(self.path, 'classifier'))

        if time_limit is None:
            time_limit = 24 * 60 * 60

        start_time = time.time()
        time_remain = time_limit
        try:
            while time_remain > 0:
                run_searcher_once(train_data, test_data, self.path, int(time_remain))
                if len(self.load_searcher().history) >= Constant.MAX_MODEL_NUM:
                    break
                time_elapsed = time.time() - start_time
                time_remain = time_limit - time_elapsed
            # if no search executed during the time_limit, then raise an error
            if time_remain <= 0:
                raise TimeoutError
        except TimeoutError:
            if len(self.load_searcher().history) == 0:
                raise TimeoutError("Search Time too short. No model was found during the search time.")
            elif self.verbose:
                print('Time is out.')

    def inverse_transform_y(self, output):
        return self.y_encoder.inverse_transform(output)

    def get_n_output_node(self):
        pass

    @property
    def metric(self):
        return Accuracy

