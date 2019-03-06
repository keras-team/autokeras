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

import numpy as np
import torch
from abc import ABC

from autokeras.constant import Constant
from autokeras.pretrained.base import Pretrained
from autokeras.text.pretrained_bert.modeling import BertForSequenceClassification
from autokeras.text.pretrained_bert.utils import convert_examples_to_features
from autokeras.text.pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


class TextClassifier(Pretrained, ABC):
    """A pre-trained TextClassifier class based on Google AI's BERT model.

    Attributes:
        model: Type of BERT model to be used for the classification task. E.g:- Uncased, Cased, etc.
        The current pre-trained models are using 'bert-base-uncased'.
        tokenizer: Tokenizer used with BERT model.
    """

    def __init__(self, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        model_state_dict = torch.load(self.local_paths[0], map_location=lambda storage, loc: storage)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                   state_dict=model_state_dict,
                                                                   num_labels=num_classes)
        self.model.to(self.device)

    def y_predict(self, x_predict):
        """ Predict the labels for the provided input data.

        Args:
            x_predict: ndarray containing the data inputs.

        Returns:
            ndarray containing the predicted labels/outputs for x_predict.
        """
        all_input_ids, all_input_mask, all_segment_ids = convert_examples_to_features([x_predict],
                                                                                      self.tokenizer,
                                                                                      max_seq_length=128)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        self.model.eval()
        for input_ids, input_mask, segment_ids in eval_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()

            for logit in logits:
                exp = np.exp(logit)
                exp = exp / np.sum(exp)
                y_pred = exp

        return y_pred


class SentimentAnalysis(TextClassifier):
    """A SentimentAnalysis class inherited from TextClassifier.

    The model is trained on the IMDb dataset. The link for the dataset is given below.
    http://ai.stanford.edu/~amaas/data/sentiment/
    """

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, **kwargs)

    @property
    def _google_drive_files(self):
        return Constant.SENTIMENT_ANALYSIS_MODELS

    def predict(self, x_predict, **kwargs):
        y_pred = self.y_predict(x_predict)
        return round(y_pred[1], 2)


class TopicClassifier(TextClassifier):
    """A pre-trained TopicClassifier class inherited from TextClassifier.

    The model is trained on the AG News dataset. The link for the dataset is given below.
    https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
    """

    def __init__(self, **kwargs):
        super().__init__(num_classes=4, **kwargs)

    @property
    def _google_drive_files(self):
        return Constant.TOPIC_CLASSIFIER_MODELS

    def predict(self, x_predict, **kwargs):
        y_pred = self.y_predict(x_predict)
        class_id = np.argmax(y_pred)
        if class_id == 0:
            return "Business"
        elif class_id == 1:
            return "Sci/Tech"
        elif class_id == 2:
            return "World"
        else:
            return "Sports"
