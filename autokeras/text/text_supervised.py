import os
import pickle
from functools import reduce

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from autokeras.constant import Constant
from autokeras.image_supervised import ImageSupervised, _validate
from autokeras.loss_function import classification_loss
from autokeras.metric import Accuracy
from autokeras.preprocessor import OneHotEncoder
from autokeras.search import Searcher
from autokeras.text.text_preprocessor import text_preprocess
from autokeras.utils import pickle_to_file


class TextDataset(Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    def __getitem__(self, index):
        if self.target is None:
            return self.dataset[index]
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.dataset)


def text_dataloader(x_data, y_data=None, batch_size=Constant.MAX_BATCH_SIZE, shuffle=True):
    x_data = torch.Tensor(x_data.transpose(0, 3, 1, 2))
    dataLoader = DataLoader(TextDataset(x_data, y_data), batch_size=batch_size, shuffle=shuffle)
    return dataLoader


class TextClassifier(ImageSupervised):
    @property
    def loss(self):
        return classification_loss

    def fit(self, x_train=None, y_train=None, batch_size=None, time_limit=None):
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
        if self.augment:
            x_train = text_preprocess(x_train, path=self.path)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        _validate(x_train, y_train)
        y_train = self.transform_y(y_train)

        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        # Create the searcher and save on disk
        if not self.searcher:
            input_shape = x_train.shape[1:]
            self.searcher_args['n_output_node'] = self.get_n_output_node()
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

        # Wrap the data into DataLoaders
        train_data = text_dataloader(x_train, y_train, batch_size=batch_size, shuffle=True)
        test_data = text_dataloader(x_test, y_test, shuffle=True)

        # Save the classifier
        pickle.dump(self, open(os.path.join(self.path, 'classifier'), 'wb'))
        pickle_to_file(self, os.path.join(self.path, 'classifier'))

        if time_limit is None:
            time_limit = 24 * 60 * 60

        self.cnn.fit(self.get_n_output_node(), x_train.shape, train_data, test_data, time_limit)

    def final_fit(self, x_train=None, y_train=None, x_test=None, y_test=None, trainer_args=None, retrain=False):
        """Final training after found the best architecture.

        Args:
            x_train: A numpy.ndarray of training data.
            y_train: A numpy.ndarray of training targets.
            x_test: A numpy.ndarray of testing data.
            y_test: A numpy.ndarray of testing targets.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
            retrain: A boolean of whether reinitialize the weights of the model.
        """
        if trainer_args is None:
            trainer_args = {'max_no_improvement_num': 30}

        if x_test is None:
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                                test_size=min(Constant.VALIDATION_SET_SIZE,
                                                                              int(len(y_train) * 0.2)),
                                                                random_state=42)
        if self.augment:
            x_train = text_preprocess(x_train, path=self.path)
            x_test = text_preprocess(x_test, path=self.path)

        y_train = self.transform_y(y_train)
        y_test = self.transform_y(y_test)

        train_data = text_dataloader(x_train, y_train, batch_size=Constant.MAX_BATCH_SIZE)
        test_data = text_dataloader(x_test, y_test, batch_size=Constant.MAX_BATCH_SIZE)

        self.cnn.final_fit(train_data, test_data, trainer_args, retrain)

    @property
    def metric(self):
        return Accuracy

    def predict(self, x_test):
        """Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the results.
        """
        if Constant.LIMIT_MEMORY:
            pass
        test_loader = text_dataloader(x_test)
        model = self.cnn.best_model
        model.eval()

        outputs = []
        with torch.no_grad():
            for index, inputs in enumerate(test_loader):
                outputs.append(model(inputs).numpy())
        output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
        return self.inverse_transform_y(output)

    def evaluate(self, x_test, y_test):
        if self.augment:
            x_test = text_preprocess(x_test, path=self.path)
        """Return the accuracy score between predict value and `y_test`."""
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_test, y_predict)

    def transform_y(self, y_train):
        # Transform y_train.
        if self.y_encoder is None:
            self.y_encoder = OneHotEncoder()
            self.y_encoder.fit(y_train)
        y_train = self.y_encoder.transform(y_train)
        return y_train

    def inverse_transform_y(self, output):
        return self.y_encoder.inverse_transform(output)

    def get_n_output_node(self):
        return self.y_encoder.n_classes
