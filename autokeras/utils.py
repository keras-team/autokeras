import os
import pickle
import torch

from torch.utils.data import DataLoader

from autokeras import constant


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
    def __init__(self, max_no_improvement_num=constant.MAX_NO_IMPROVEMENT_NUM, min_loss_dec=constant.MIN_LOSS_DEC):
        super().__init__()
        self.training_losses = []
        self.minimum_loss = None
        self._no_improvement_count = 0
        self._max_no_improvement_num = max_no_improvement_num
        self._done = False
        self._min_loss_dec = min_loss_dec
        self.max_accuracy = 0

    def on_train_begin(self, logs=None):
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
    """A class that is used to train model

    This class can train a model with dataset and will not stop until getting minimum loss

    Attributes:
        model: the model that will be trained
        x_train: the input train data
        y_train: the input train data labels
        x_test: the input test data
        y_test: the input test data labels
        verbose: verbosity mode
    """

    def __init__(self, model, train_data, test_data, verbose):
        """Init ModelTrainer with model, x_train, y_train, x_test, y_test, verbose"""
        self.model = model
        self.verbose = verbose
        self.train_data = train_data
        self.test_data = test_data
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr_schedule(0), momentum=0.9, weight_decay=5e-4)
        self.early_stop = None

    def train_model(self,
                    max_iter_num=constant.MAX_ITER_NUM,
                    max_no_improvement_num=constant.MAX_NO_IMPROVEMENT_NUM,
                    batch_size=constant.MAX_BATCH_SIZE):
        """Train the model.

        Args:
            max_iter_num: An integer. The maximum number of epochs to train the model.
                The training will stop when this number is reached.
            max_no_improvement_num: An integer. The maximum number of epochs when the loss value doesn't decrease.
                The training will stop when this number is reached.
            batch_size: An integer. The batch size during the training.
            optimizer: An optimizer class.
        """
        batch_size = min(len(self.train_data), batch_size)

        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True)

        self.early_stop = EarlyStop(max_no_improvement_num)
        self.early_stop.on_train_begin()

        for epoch in range(max_iter_num):
            self._train(train_loader)
            test_loss = self._test(test_loader)
            terminate = self.early_stop.on_epoch_end(test_loss)
            if terminate:
                break

    def _train(self, loader):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            targets = targets.argmax(1)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    def _test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                targets = targets.argmax(1)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return test_loss


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
