from abc import ABC

import numpy as np
import os
import torch

from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.supervised import SingleModelSupervised
from autokeras.text.pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from autokeras.text.pretrained_bert.modeling import BertForSequenceClassification
from autokeras.text.pretrained_bert.optimization import BertAdam
from autokeras.text.pretrained_bert.tokenization import BertTokenizer
from autokeras.utils import get_device, temp_path_generator
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def get_inputs(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class TextClassifier(SingleModelSupervised, ABC):
    """TextClassifier class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = get_device()

        # BERT specific
        self.bert_model = 'bert-base-uncased'
        self.max_seq_length = 128
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)

        # Labels/classes
        self.label_list = None
        self.num_labels = None

        # Output directory
        self.output_dir = temp_path_generator() + 'bert_classifier/'
        self.output_model_file = self.output_dir + 'pytorch_model.bin'

        # Training params
        self.global_step = 0
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.nb_tr_steps = 1
        self.num_train_epochs = 4
        self.tr_loss = 0
        self.train_batch_size = 32
        self.warmup_proportion = 0.1

        # Evaluation params
        self.eval_batch_size = 32

    def fit(self, x, y, time_limit=None):
        if os.path.exists(self.output_dir) and os.listdir(self.output_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)

        self.label_list = list(set(y))
        self.num_labels = len(self.label_list)

        num_train_steps = int(
            len(x) / self.train_batch_size / self.gradient_accumulation_steps * self.num_train_epochs)

        # Prepare model
        model = BertForSequenceClassification.from_pretrained(self.bert_model,
                                                              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE/'distributed_-1',
                                                              num_labels=self.num_labels)

        model.to(self.device)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps

        # Add bert adam
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.learning_rate,
                             warmup=self.warmup_proportion,
                             t_total=t_total)

        train_features = self.preprocess(x)
        print("***** Running training *****")
        print("  Num examples = %d", len(x))
        print("  Batch size = %d", self.train_batch_size)
        print("  Num steps = %d", num_train_steps)
        all_input_ids, all_input_mask, all_segment_ids = get_inputs(train_features)
        all_label_ids = torch.tensor([int(f) for f in y], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)

        model.train()
        for _ in trange(int(self.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = self.learning_rate * warmup_linear(self.global_step / t_total,
                                                                           self.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    self.global_step += 1

        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), self.output_model_file)

    def predict(self, x_test):
        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(self.output_model_file)
        model = BertForSequenceClassification.from_pretrained(self.bert_model, state_dict=model_state_dict,
                                                              num_labels=self.num_labels)
        model.to(self.device)
        eval_features = self.preprocess(x_test)
        print("***** Running evaluation *****")
        print("  Num examples = %d", len(x_test))
        print("  Batch size = %d", self.eval_batch_size)
        all_input_ids, all_input_mask, all_segment_ids = get_inputs(eval_features)
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
        return features

    def transform_y(self, y):
        pass

    def inverse_transform_y(self, output):
        return np.argmax(output, axis=1)
