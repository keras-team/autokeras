import os
import tempfile

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from autokeras.pretrained.base import Pretrained
from autokeras.text.pretrained_bert.modeling import BertForSequenceClassification
from autokeras.text.pretrained_bert.tokenization import BertTokenizer
from autokeras.utils import download_file_from_google_drive, get_device

TEXT_SENTIMENT_FILE_ID = '15kIuZrzWdoEpmZ842ufZHm3B3QZFpfLu'


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_examples_to_features(examples, max_seq_length, tokenizer):

    features = []
    for (_, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

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


class SentimentAnalysis(Pretrained):

    def __init__(self):

        super(SentimentAnalysis, self).__init__()
        self.device = None
        self.tokenizer = None
        self.model = None
        self.load()

    def load(self):

        self.device = get_device()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        output_model_file = os.path.join(tempfile.gettempdir(), 'text_sentiment_pytorch_model.bin')

        download_file_from_google_drive(TEXT_SENTIMENT_FILE_ID, output_model_file)

        model_state_dict = torch.load(output_model_file, map_location=lambda storage, loc: storage)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', state_dict=model_state_dict)
        self.model.to(self.device)

    def predict(self, x_predict):

        eval_features = convert_examples_to_features([x_predict], 128, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        self.model.eval()
        sentence_polarity = None
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
                sentence_polarity = round(exp[1], 2)

        return sentence_polarity
