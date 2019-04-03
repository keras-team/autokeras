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
from functools import reduce

import os
import torch
from copy import deepcopy
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm, trange

from autokeras.constant import Constant
from autokeras.nn.model_trainer import ModelTrainerBase
from autokeras.text.pretrained_bert.optimization import BertAdam, warmup_linear


def get_device():
    """ If CUDA is available, use CUDA device, else use CPU device.
    Returns: string device name
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


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
        if self.device is None:
            self.device = get_device()
        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.optimizer = None
        self.early_stop = None
        self.scheduler = None
        self.current_epoch = 0
        self.current_metric_value = 0
        self.temp_model_path = os.path.join(path, 'temp_model')

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

        self.early_stop = EarlyStop(max_no_improvement_num)
        self.early_stop.on_train_begin()
        self._timeout = time.time() + timeout if timeout is not None else sys.maxsize

        test_metric_value_list = []
        test_loss_list = []
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=3e-4)
        # self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, max_iter_num)

        for epoch in range(max_iter_num):
            self.scheduler.step()
            self._train()
            test_loss, metric_value = self._test()
            self.current_metric_value = metric_value
            test_metric_value_list.append(metric_value)
            test_loss_list.append(test_loss)
            decreasing = self.early_stop.on_epoch_end(test_loss)

            if self.early_stop.no_improvement_count == 0:
                self._save_model()

            if not decreasing:
                if self.verbose:
                    print('\nNo loss decrease after {} epochs.\n'.format(max_no_improvement_num))
                self._load_model()
                break

        last_num = min(max_no_improvement_num, max_iter_num)
        return (sum(test_loss_list[-last_num:]) / last_num,
                sum(test_metric_value_list[-last_num:]) / last_num)

    def _train(self):
        """Where the actual train proceed."""
        self.model.train()
        loader = self.train_loader
        self.current_epoch += 1

        if self.verbose:
            progress_bar = self.init_progress_bar(len(loader))
        else:
            progress_bar = None

        for batch_idx, (inputs, targets) in enumerate(deepcopy(loader)):
            if time.time() >= self._timeout:
                raise TimeoutError
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if self.verbose:
                if batch_idx % 10 == 0:
                    progress_bar.update(10)
        if self.verbose:
            progress_bar.close()

    def _test(self):
        """Function for evaluation."""
        self.model.eval()
        test_loss = 0
        all_targets = []
        all_predicted = []
        loader = self.test_loader

        if self.verbose:
            progress_bar = self.init_progress_bar(len(loader))
        else:
            progress_bar = None

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(deepcopy(loader)):
                if time.time() >= self._timeout:
                    raise TimeoutError
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                # cast tensor to float
                test_loss += float(self.loss_function(outputs, targets))

                all_predicted.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                if self.verbose:
                    if batch_idx % 10 == 0:
                        progress_bar.update(10)

        if self.verbose:
            progress_bar.close()

        all_predicted = reduce(lambda x, y: np.concatenate((x, y)), all_predicted)
        all_targets = reduce(lambda x, y: np.concatenate((x, y)), all_targets)
        return test_loss, self.metric.compute(all_predicted, all_targets)

    def _save_model(self):
        torch.save(self.model.state_dict(), self.temp_model_path)

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.temp_model_path))

    def init_progress_bar(self, loader_len):
        return tqdm(total=loader_len,
                    desc='Epoch-'
                         + str(self.current_epoch)
                         + ', Current Metric - '
                         + str(self.current_metric_value),
                    file=sys.stdout,
                    leave=False,
                    ncols=100,
                    position=0,
                    unit=' batch')


class EarlyStop:
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

    def on_train_begin(self):
        """Initiate the early stop condition.
        Call on every time the training iteration begins.
        """
        self.training_losses = []
        self.no_improvement_count = 0
        self._done = False
        self.minimum_loss = float('inf')

    def on_epoch_end(self, loss):
        """Check the early stop condition.
        Call on every time the training iteration end.
        Args:
            loss: The loss function achieved by the epoch.
        Returns:
            True if condition met, otherwise False.
        """
        self.training_losses.append(loss)
        if self._done and loss > (self.minimum_loss - self._min_loss_dec):
            return False

        if loss > (self.minimum_loss - self._min_loss_dec):
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
            self.minimum_loss = loss

        if self.no_improvement_count > self._max_no_improvement_num:
            self._done = True

        return True


class BERTTrainer(ModelTrainerBase):
    """A ModelTrainer for the Google AI's BERT model. Currently supports only classification task.

    Attributes:
        model: Type of BERT model to be used for the task. E.g:- Uncased, Cased, etc.
        output_model_file: File location to save the trained model.
        num_labels: Number of output labels for the classification task.
    """

    def __init__(self, train_data, model, output_model_file, loss_function=None):
        """Initialize the BERTTrainer.

        Args:
            train_data: the training data.
            model: Type of BERT model to be used for the task. E.g:- Uncased, Cased, etc.
            output_model_file: File location to save the trained model.
            num_labels: Number of output labels for the classification task.
            loss_function: The loss function for the classifier.
        """
        super().__init__(loss_function, train_data, verbose=True)

        self.train_data = train_data
        self.model = model
        self.output_model_file = output_model_file

        # Training params
        self.global_step = 0
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.nb_tr_steps = 1
        self.num_train_epochs = Constant.BERT_TRAINER_EPOCHS
        self.tr_loss = 0
        self.train_batch_size = Constant.BERT_TRAINER_BATCH_SIZE
        self.warmup_proportion = 0.1
        self.train_data_size = self.train_data.__len__()
        self.num_train_steps = int(self.train_data_size /
                                   self.train_batch_size /
                                   self.gradient_accumulation_steps *
                                   self.num_train_epochs)

    def train_model(self,
                    max_iter_num=None,
                    max_no_improvement_num=None,
                    timeout=None):
        """Train the model.

        Train the model with max_iter_num.

        Args:
            timeout: timeout in seconds
            max_iter_num: An integer. The maximum number of epochs to train the model.
            max_no_improvement_num: An integer. The maximum number of epochs when the loss value doesn't decrease.

        Returns:
            Training loss.
        """
        if max_iter_num is not None:
            self.num_train_epochs = max_iter_num

        self.model.to(self.device)

        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # Add bert adam
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.learning_rate,
                             warmup=self.warmup_proportion,
                             t_total=self.num_train_steps)

        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.train_batch_size)

        if self.verbose:
            print("***** Running training *****")
            print("Num examples = %d", self.train_data_size)
            print("Batch size = %d", self.train_batch_size)
            print("Num steps = %d", self.num_train_steps)

        self.model.train()
        for _ in trange(int(self.num_train_epochs), desc="Epoch"):
            tr_loss = self._train(optimizer, train_dataloader)

        if self.verbose:
            print("Training loss = %d", tr_loss)

        self._save_model()
        return tr_loss

    def _train(self, optimizer, dataloader):
        """ Actual training is performed here."""
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = self.model(input_ids, segment_ids, input_mask, label_ids)
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = self.learning_rate * warmup_linear(self.global_step / self.num_train_steps,
                                                                  self.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                self.global_step += 1

        return tr_loss

    def _save_model(self):
        """Save the trained model to disk."""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model
        torch.save(model_to_save.state_dict(), self.output_model_file)
