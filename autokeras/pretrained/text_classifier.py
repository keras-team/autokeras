import numpy as np
import os
import tempfile
import torch
from abc import ABC

from autokeras.constant import Constant
from autokeras.pretrained.base import Pretrained
from autokeras.text.pretrained_bert.tokenization import BertTokenizer
from autokeras.text.pretrained_bert.modeling import BertForSequenceClassification
from autokeras.utils import download_file_from_google_drive, get_device
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class TextClassifier(Pretrained, ABC):

    def __init__(self, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        model_state_dict = torch.load(self.local_paths[0], map_location=lambda storage, loc: storage)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                   state_dict=model_state_dict,
                                                                   num_labels=num_classes)
        self.model.to(self.device)

    def convert_examples_to_features(self, examples, max_seq_length):
        features = []
        for (_, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example)

            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            if len(input_ids) != max_seq_length or len(input_mask) != max_seq_length or len(segment_ids) != max_seq_length:
                raise AssertionError()

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids))
        return features

    def y_predict(self, x_predict):
        eval_features = self.convert_examples_to_features([x_predict], 128)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

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

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, **kwargs)

    @property
    def _google_drive_files(self):
        return Constant.SENTIMENT_ANALYSIS_MODELS

    def predict(self, x_predict, **kwargs):
        y_pred = self.y_predict(x_predict)
        return round(y_pred[1], 2)


class TopicClassifier(TextClassifier):

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

