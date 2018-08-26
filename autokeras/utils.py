import os
import pickle
import sys
import tempfile
from copy import deepcopy
from functools import reduce

import numpy as np
import torch

from autokeras.constant import Constant
from tqdm.autonotebook import tqdm

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

    This class can train a Pytorch model with the given data loaders.
    The metric, loss_function, and model must be compatible with each other.
    Please see the details in the Attributes.

    Attributes:
        device: A string. Indicating the device to use. 'cuda' or 'cpu'.
        model: An instance of Pytorch Module. The model that will be trained.
        train_loader: Training data wrapped in batches in Pytorch Dataloader.
        test_loader: Testing data wrapped in batches in Pytorch Dataloader.
        loss_function: A function with two parameters (prediction, target).
            There is no specific requirement for the types of the parameters,
            as long as they are compatible with the model and the data loaders.
            The prediction should be the output of the model for a batch.
            The target should be a batch of targets packed in the data loaders.
        optimizer: The optimizer is chosen to use the Pytorch Adam optimizer.
        early_stop: An instance of class EarlyStop.
        metric: It should be a subclass of class autokeras.metric.Metric.
            In the compute(prediction, target) function, prediction and targets are
            all numpy arrays converted from the output of the model and the targets packed in the data loaders.
        verbose: Verbosity mode.
    """

    def __init__(self, model, train_loader, test_loader, metric, loss_function, verbose):
        """Init the ModelTrainer with `model`, `x_train`, `y_train`, `x_test`, `y_test`, `verbose`"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model.to(self.device)
        self.verbose = verbose
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_function = loss_function
        self.optimizer = None
        self.early_stop = None
        self.metric = metric

    def train_model(self,
                    max_iter_num=None,
                    max_no_improvement_num=None):
        """Train the model.

        Args:
            max_iter_num: An integer. The maximum number of epochs to train the model.
                The training will stop when this number is reached.
            max_no_improvement_num: An integer. The maximum number of epochs when the loss value doesn't decrease.
                The training will stop when this number is reached.
        """
        if max_iter_num is None:
            max_iter_num = Constant.MAX_ITER_NUM

        if max_no_improvement_num is None:
            max_no_improvement_num = Constant.MAX_NO_IMPROVEMENT_NUM

        self.early_stop = EarlyStop(max_no_improvement_num)
        self.early_stop.on_train_begin()

        test_metric_value_list = []
        test_loss_list = []
        self.optimizer = torch.optim.Adam(self.model.parameters())

        if self.verbose:
            pbar = tqdm(total=max_iter_num,
                        desc='    Model    ',
                        file=sys.stdout,
                        leave=False,
                        ncols=75,
                        position=1,
                        unit=' epoch')

        for epoch in range(max_iter_num):
            self._train()
            test_loss, metric_value = self._test()
            test_metric_value_list.append(metric_value)
            test_loss_list.append(test_loss)
            if self.verbose:
                pbar.update(1)
                if epoch == 0:
                    header = ['Epoch', 'Loss', 'Accuracy']
                    line = '|'.join(x.center(24) for x in header)
                    pbar.write('+' + '-' * len(line) + '+')
                    pbar.write('|' + line + '|')
                    pbar.write('+' + '-' * len(line) + '+')
                r = [epoch + 1, test_loss, metric_value]
                line = '|'.join(str(x).center(24) for x in r)
                pbar.write('|' + line + '|')
                pbar.write('+' + '-' * len(line) + '+')
            decreasing = self.early_stop.on_epoch_end(test_loss)
            if not decreasing:
                if self.verbose:
                    print('\nNo loss decrease after {} epochs.\n'.format(max_no_improvement_num))
                break
        if self.verbose:
            pbar.close()
        return (sum(test_loss_list[-max_no_improvement_num:]) / max_no_improvement_num,
                sum(test_metric_value_list[-max_no_improvement_num:]) / max_no_improvement_num)

    def _train(self):
        self.model.train()
        loader = self.train_loader

        cp_loader = deepcopy(loader)
        if self.verbose:
            pbar = tqdm(total=len(cp_loader),
                        desc='Current Epoch',
                        file=sys.stdout,
                        leave=False,
                        ncols=75,
                        position=0,
                        unit=' batch')

        for batch_idx, (inputs, targets) in enumerate(cp_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if self.verbose:
                if batch_idx % 10 == 0:
                    pbar.update(10)
        if self.verbose:
            pbar.close()

    def _test(self):
        self.model.eval()
        test_loss = 0
        all_targets = []
        all_predicted = []
        loader = self.test_loader
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(deepcopy(loader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                # cast tensor to float
                test_loss += float(self.loss_function(outputs, targets))

                all_predicted.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        all_predicted = reduce(lambda x, y: np.concatenate((x, y)), all_predicted)
        all_targets = reduce(lambda x, y: np.concatenate((x, y)), all_targets)
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


def temp_folder_generator():
    sys_temp = tempfile.gettempdir()
    path = os.path.join(sys_temp, 'autokeras')
    if not os.path.exists(path):
        os.makedirs(path)
    return path
