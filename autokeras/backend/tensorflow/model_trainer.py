# coding=utf-8
# Original work Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Modified work Copyright 2019 The AutoKeras team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time

import numpy as np
import os
from tensorflow.keras import optimizers, models, metrics
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ReduceLROnPlateau
from autokeras.constant import Constant
from autokeras.nn.model_trainer import ModelTrainerBase


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


class EarlyStop(Callback):
    """A class check for early stop condition.
    Attributes:
        training_losses: Record all the training loss.
        minimum_loss: The minimum loss we achieve so far. Used to compared to determine no improvement condition.
        no_improvement_count: Current no improvement count.
        _max_no_improvement_num: The maximum number specified.
        _done: Whether condition met.
        _min_loss_dec: A threshold for loss improvement.
    """

    def __init__(self, max_no_improvement_num=None, min_loss_dec=None):
        super().__init__()
        self.training_losses = []
        self.minimum_loss = None
        self.no_improvement_count = 0
        self._max_no_improvement_num = max_no_improvement_num if max_no_improvement_num is not None \
            else Constant.MAX_NO_IMPROVEMENT_NUM
        self._done = False
        self._min_loss_dec = min_loss_dec if min_loss_dec is not None else Constant.MIN_LOSS_DEC
        self.max_accuracy = 0

    def on_train_begin(self, logs=None):
        """Initiate the early stop condition.
        Call on every time the training iteration begins.
        """
        self.training_losses = []
        self.no_improvement_count = 0
        self._done = False
        self.minimum_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        """Check the early stop condition.
        Call on every time the training iteration end.
        Args:
            loss: The loss function achieved by the epoch.
        Returns:
            True if condition met, otherwise False.
        """
        # self.max_accuracy = max(self.max_accuracy, logs.get('val_acc'))
        self.max_accuracy = logs.get('val_acc')
        loss = logs.get('val_loss')
        self.training_losses.append(loss)
        if self._done and loss > (self.minimum_loss - self._min_loss_dec):
            raise NoImprovementError('No improvement for {} epochs.'.format(self._max_no_improvement_num))

        if loss > (self.minimum_loss - self._min_loss_dec):
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
            self.minimum_loss = loss

        if self.no_improvement_count > self._max_no_improvement_num:
            self._done = True


class ModelTrainer(ModelTrainerBase):
    """A class that is used to train the model.
    This class can train a Pytorch model with the given data loaders.
    The metric, loss_function, and model must be compatible with each other.
    Please see the details in the Attributes.
    Attributes:
        temp_model_path: Specify the path where temp model should be stored.
        model: An instance of Pytorch Module. The model that will be trained.
        early_stop: An instance of class EarlyStop.
        optimizer: The optimizer is chosen to use the Pytorch Adam optimizer.
        current_epoch: Record the current epoch.
    """

    def __init__(self, model, path, **kwargs):
        super().__init__(**kwargs)
        self.model = model.model
        # TODO: TF Parallel
        self.optimizer = None
        self.early_stop = None
        self.scheduler = None
        self.reducer = None
        self.current_epoch = 0
        self.current_metric_value = 0
        self.temp_model_path = os.path.join(path, 'temp_model.h5')

        # TODO: better way to define keras metrics
        if self.loss_function.__name__ == 'classification_loss':
            self.keras_metric = metrics.categorical_accuracy
        elif self.loss_function.__name__ == 'regression_loss':
            self.keras_metric = metrics.mean_squared_error
        elif self.loss_function.__name__ == 'binary_classification_loss':
            self.keras_metric = metrics.binary_accuracy

    def train_model(self,
                    lr=0.001,
                    max_iter_num=None,
                    max_no_improvement_num=None,
                    timeout=None):
        """Train the model.
        Train the model with max_iter_num or max_no_improvement_num is met.
        Args:
            lr: learning rate of the traininig
            timeout: timeout in seconds
            max_iter_num: An integer. The maximum number of epochs to train the model.
                The training will stop when this number is reached.
            max_no_improvement_num: An integer. The maximum number of epochs when the loss value doesn't decrease.
                The training will stop when this number is reached.
        Returns:
            A tuple of loss values and metric value.
        """
        if max_iter_num is None:
            max_iter_num = Constant.MAX_ITER_NUM

        if max_no_improvement_num is None:
            max_no_improvement_num = Constant.MAX_NO_IMPROVEMENT_NUM

        # callback functions
        self.early_stop = EarlyStop(max_no_improvement_num)
        self.scheduler = LearningRateScheduler(lr_schedule)
        self.reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        callbacks = [self.early_stop, self.scheduler, self.reducer]

        # customize optimizer and compile model
        self.optimizer = optimizers.SGD(lr=lr, momentum=0.9, decay=3e-4)  # clipvalue=1.0,
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=[self.keras_metric])

        # fit model
        # TODO: raise TimeoutError when timeout
        self._timeout = time.time() + timeout if timeout is not None else sys.maxsize
        try:
            self.model.fit_generator(self.train_loader,
                                     epochs=max_iter_num,
                                     validation_data=self.test_loader,
                                     callbacks=callbacks,
                                     verbose=self.verbose)
        except NoImprovementError as e:
            if self.verbose:
                print('Training finished!')
                print(e.message)
            return self.early_stop.minimum_loss, self.early_stop.max_accuracy
        return self.early_stop.minimum_loss, self.early_stop.max_accuracy

    def _save_model(self):
        self.model.save(self.temp_model_path)

    def _load_model(self):
        self.model = models.load_model(self.temp_model_path)
