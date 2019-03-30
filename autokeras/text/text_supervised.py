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

from abc import ABC, abstractmethod

import numpy as np
import os
import torch

from autokeras.backend import Backend
from autokeras.nn.metric import Accuracy
from autokeras.backend.torch.model_trainer import BERTTrainer
from autokeras.supervised import SingleModelSupervised
from autokeras.text.pretrained_bert.utils import PYTORCH_PRETRAINED_BERT_CACHE
from autokeras.text.pretrained_bert.modeling import BertForSequenceClassification
from autokeras.text.pretrained_bert.utils import convert_examples_to_features
from autokeras.text.pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


class TextSupervised(SingleModelSupervised, ABC):

    def __init__(self, verbose, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.device = Backend.get_device()

        # BERT specific
        self.bert_model = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)

        # Labels/classes
        self.num_labels = None

    @abstractmethod
    def fit(self, x, y, time_limit=None):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @property
    @abstractmethod
    def metric(self):
        pass

    @property
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    def transform_y(self, y):
        pass

    def inverse_transform_y(self, output):
        return np.argmax(output, axis=1)


class TextRegressor(TextSupervised):
    pass


class TextClassifier(TextSupervised):
    """A TextClassifier class based on Google AI's BERT model.

    Attributes:
        device: Specific hardware for using/running the model. E.g:- CPU, GPU or TPU.
        verbose: Mode of verbosity.
        bert_model: Type of BERT model to be used for the classification task. E.g:- Uncased, Cased, etc.
        tokenizer: Tokenizer used with BERT model.
        num_labels: Number of output labels for the classification task.
        output_model_file: File location to save the trained model.
    """

    def __init__(self, verbose, **kwargs):
        """Initialize the TextClassifier.

        Args:
            verbose: Mode of verbosity.
        """
        super().__init__(verbose=verbose, **kwargs)
        self.device = Backend.get_device()
        self.verbose = verbose

        # BERT specific
        self.bert_model = 'bert-base-uncased'
        self.max_seq_length = 128
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)

        # Labels/classes
        self.num_labels = None

        # Output directory
        self.output_model_file = os.path.join(self.path, 'pytorch_model.bin')

        # Evaluation params
        self.eval_batch_size = 32

    def fit(self, x, y, time_limit=None):
        """ Train the text classifier based on the training data.

        Args:
            x: ndarray containing the train data inputs.
            y: ndarray containing the train data outputs/labels.
            time_limit: Maximum time allowed for searching. It does not apply for this classifier.
        """
        self.num_labels = len(list(set(y)))

        # Prepare model
        model = BertForSequenceClassification.from_pretrained(self.bert_model,
                                                              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE/'distributed_-1',
                                                              num_labels=self.num_labels)

        all_input_ids, all_input_mask, all_segment_ids = self.preprocess(x)
        all_label_ids = torch.tensor([int(f) for f in y], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        bert_trainer = BERTTrainer(train_data, model, self.output_model_file, self.num_labels)
        bert_trainer.train_model()

    def predict(self, x_test):
        """ Predict the labels for the provided input data.

        Args:
            x_test: ndarray containing the test data inputs.

        Returns:
            ndarray containing the predicted labels/outputs for x_test.
        """
        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(self.output_model_file)
        model = BertForSequenceClassification.from_pretrained(self.bert_model, state_dict=model_state_dict,
                                                              num_labels=self.num_labels)
        model.to(self.device)

        if self.verbose:
            print("***** Running evaluation *****")
            print("  Num examples = %d", len(x_test))
            print("  Batch size = %d", self.eval_batch_size)
        all_input_ids, all_input_mask, all_segment_ids = self.preprocess(x_test)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)

        model.eval()
        y_preds = []
        for input_ids, input_mask, segment_ids in eval_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            y_preds.extend(logits)
        return self.inverse_transform_y(y_preds)

    @property
    def metric(self):
        return Accuracy

    @property
    def loss(self):
        return Backend.classification_loss

    def preprocess(self, x):
        """ Preprocess text data.

        Tokenize the input text and convert into features.

        Args:
            x: Text input.

        Returns:
            all_input_ids: ndarray containing the ids for each token.
            all_input_masks: ndarray containing 1's or 0's based on if the tokens are real or padded.
            all_segment_ids: ndarray containing all 0's since it is a classification task.
        """
        return convert_examples_to_features(x, self.tokenizer, self.max_seq_length)

    def transform_y(self, y):
        pass

    def inverse_transform_y(self, output):
        return np.argmax(output, axis=1)
