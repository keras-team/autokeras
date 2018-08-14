import os
import sys
import pickle
import numpy as np
from copy import deepcopy

import torch

from torch.utils.data import DataLoader

from autokeras.constant import Constant


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


class NoImprovementError(Exception):
    def __init__(self, message):
        self.message = message


class EarlyStop:
    def __init__(self, max_no_improvement_num=Constant.MAX_NO_IMPROVEMENT_NUM, min_loss_dec=Constant.MIN_LOSS_DEC):
        super().__init__()
        self.training_losses = []
        self.minimum_loss = None
        self._no_improvement_count = 0
        self._max_no_improvement_num = max_no_improvement_num
        self._done = False
        self._min_loss_dec = min_loss_dec
        self.max_accuracy = 0

    def on_train_begin(self):
        self.training_losses = []
        self._no_improvement_count = 0
        self._done = False
        self.minimum_loss = float('inf')

    def on_epoch_end(self, loss):
        self.training_losses.append(loss)
        if self._done and loss > (self.minimum_loss - self._min_loss_dec):
            return False

        if loss > (self.minimum_loss - self._min_loss_dec):
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
            self.minimum_loss = loss

        if self._no_improvement_count > self._max_no_improvement_num:
            self._done = True

        return True


class ModelTrainer:
    """A class that is used to train the model.

    This class can train a model with dataset and will not stop until getting the minimum loss.

    Attributes:
        model: The model that will be trained
        train_data: Training data wrapped in batches.
        test_data: Testing data wrapped in batches.
        verbose: Verbosity mode.
    """

    def __init__(self, model, train_data, test_data, metric, verbose):
        """Init the ModelTrainer with `model`, `x_train`, `y_train`, `x_test`, `y_test`, `verbose`"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model.to(self.device)
        self.verbose = verbose
        self.train_data = train_data
        self.test_data = test_data
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = None
        self.early_stop = None
        self.metric = metric

    def train_model(self,
                    max_iter_num=None,
                    max_no_improvement_num=None,
                    batch_size=None):
        """Train the model.

        Args:
            max_iter_num: An integer. The maximum number of epochs to train the model.
                The training will stop when this number is reached.
            max_no_improvement_num: An integer. The maximum number of epochs when the loss value doesn't decrease.
                The training will stop when this number is reached.
            batch_size: An integer. The batch size during the training.
        """
        if max_iter_num is None:
            max_iter_num = Constant.MAX_ITER_NUM

        if max_no_improvement_num is None:
            max_no_improvement_num = Constant.MAX_NO_IMPROVEMENT_NUM

        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(self.train_data), batch_size)

        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True)

        self.early_stop = EarlyStop(max_no_improvement_num)
        self.early_stop.on_train_begin()

        test_accuracy_list = []
        test_loss_list = []
        self.optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in range(max_iter_num):
            self._train(train_loader, epoch)
            test_loss, accuracy = self._test(test_loader)
            test_accuracy_list.append(accuracy)
            test_loss_list.append(test_loss)
            if self.verbose:
                print('Epoch {}: loss {}, accuracy {}'.format(epoch + 1, test_loss, accuracy))
            decreasing = self.early_stop.on_epoch_end(test_loss)
            if not decreasing:
                if self.verbose:
                    print('No loss decrease after {} epochs'.format(max_no_improvement_num))
                break
        return (sum(test_loss_list[-max_no_improvement_num:]) / max_no_improvement_num,
                sum(test_accuracy_list[-max_no_improvement_num:]) / max_no_improvement_num)

    def _train(self, loader, epoch):
        self.model.train()

        for batch_idx, (inputs, targets) in enumerate(deepcopy(loader)):
            targets = targets.argmax(1)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = torch.nn.functional.nll_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if self.verbose:
                if batch_idx % 10 == 0:
                    print('.', end='')
                    sys.stdout.flush()
        if self.verbose:
            print()

    def _test(self, test_loader):
        self.model.eval()
        test_loss = 0
        all_targets = []
        all_predicted = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(deepcopy(test_loader)):
                targets = targets.argmax(1)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets)

                _, predicted = outputs.max(1)
                all_predicted = np.concatenate((all_predicted, predicted.numpy()))
                all_targets = np.concatenate((all_targets, targets.numpy()))
        return test_loss, self.metric.compute(all_predicted, all_targets)


def ensure_dir(directory):
    """Create directory if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_file_dir(path):
    """Create path if it does not exist"""
    ensure_dir(os.path.dirname(path))


def has_file(path):
    return os.path.exists(path)


def pickle_from_file(path):
    return pickle.load(open(path, 'rb'))


def pickle_to_file(obj, path):
    pickle.dump(obj, open(path, 'wb'))
