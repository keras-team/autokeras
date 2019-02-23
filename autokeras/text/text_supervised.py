from abc import ABC

import numpy as np
import os
import torch

from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.nn.model_trainer import BERTTrainer
from autokeras.supervised import SingleModelSupervised
from autokeras.text.pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from autokeras.text.pretrained_bert.modeling import BertForSequenceClassification
from autokeras.text.pretrained_bert.tokenization import BertTokenizer
from autokeras.utils import get_device, temp_path_generator
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class TextClassifier(SingleModelSupervised, ABC):
    """TextClassifier class.
    """

    def __init__(self, verbose, **kwargs):
        super().__init__(**kwargs)
        self.device = get_device()
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
        return classification_loss

    def preprocess(self, x):
        features = []
        for (_, example) in enumerate(x):
            tokens_a = self.tokenizer.tokenize(example)

            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            if len(input_ids) != self.max_seq_length or \
                    len(input_mask) != self.max_seq_length or \
                    len(segment_ids) != self.max_seq_length:
                raise AssertionError()

            features.append(InputFeatures(input_ids=input_ids,
                                          input_mask=input_mask,
                                          segment_ids=segment_ids))

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids

    def transform_y(self, y):
        pass

    def inverse_transform_y(self, output):
        return np.argmax(output, axis=1)
