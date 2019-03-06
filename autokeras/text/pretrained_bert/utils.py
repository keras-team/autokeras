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

import os
import torch

from pathlib import Path

from autokeras.utils import download_file_from_google_drive

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """ Convert text examples to BERT specific input format.

    Tokenize the input text and convert into features.

    Args:
        examples: Text data.
        tokenizer: Tokenizer to process the text into tokens.
        max_seq_length: The maximum length of the text sequence supported.

    Returns:
        all_input_ids: ndarray containing the ids for each token.
        all_input_masks: ndarray containing 1's or 0's based on if the tokens are real or padded.
        all_segment_ids: ndarray containing all 0's since it is a classification task.
    """
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

        if len(input_ids) != max_seq_length or \
                len(input_mask) != max_seq_length or \
                len(segment_ids) != max_seq_length:
            raise AssertionError()

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    return all_input_ids, all_input_mask, all_segment_ids


def cached_path(file_info, cache_dir=None):
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE

    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, file_info.local_name)

    if not os.path.exists(file_path):
        download_file_from_google_drive(file_id=file_info.google_drive_id,
                                        dest_path=file_path,
                                        verbose=True)
    return file_path
